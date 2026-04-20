# =============================================================================
# graphene_qtci_all.jl
# =============================================================================
# Archivo único que unifica:
#   - Parte ij  : QTCI con dimensión local 4, codificación (bit_row, bit_col)
#   - Parte ijkl: QTCI con dimensión local 16, codificación (bi, bj, bk, bl)
#   - MPO exacto sparse → quantics vía autómata DAG de prefijos/sufijos
#   - AutoMPO vía ITensors (representación many-body)
#   - Patching pQTCI adaptativo fiel al paper arXiv:2602.22372
#   - Hamiltoniano de torus periódico (sin contorno) para comparación
#
# CONVENCIONES:
#   W[ℓ] :: Array{T,4}  con índices (bondL, rowbit, colbit, bondR)   [parte ij]
#   W[ℓ] :: Array{T,6}  con índices (bondL, bi, bj, bk, bl, bondR)  [parte ijkl]
#   El Hamiltoniano del flake usa Quantica: red honeycomb + supercell estrellada.
#   El torus usa condiciones de contorno periódicas puras (sin flake).
#
# NOTAS IMPORTANTES SOBRE KERNEL:
#   - Nunca guardes grandes objetos en variables globales; usa variables locales.
#   - Llama GC.gc() explícitamente en barridos.
#   - Las validaciones pesadas (validate_all_nonzeros) están off por defecto.
#   - El notebook usa parámetros "quick" por defecto; sube tol/maxbonddim a mano.
# =============================================================================

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Quantica
using Plots
import TensorCrossInterpolation as TCI

# ITensors/ITensorMPS son opcionales; se cargan solo si están disponibles
const _HAVE_ITENSORS = try
    using ITensors, ITensorMPS
    true
catch
    false
end

# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================

const DEFAULT_TOL        = 1e-13
const DEFAULT_MAXBONDDIM = 512
const DEFAULT_MAXITER    = 20
const DEFAULT_SEED       = 1234

# =============================================================================
# SECCIÓN 1 — HAMILTONIANO SPARSE (flake con contorno estrellado)
# =============================================================================

"""
    build_graphene_hamiltonian_sparse(; radius, star_amp)

Devuelve (h, Hs, N) donde:
- h   : objeto Quantica Hamiltonian
- Hs  : SparseMatrixCSC del Hamiltoniano de tight-binding
- N   : número de sitios

Contorno: región estrellada de 5 puntas. Con star_amp=0 el contorno es circular.
El hopping es t=1 entre primeros vecinos en la red honeycomb.
"""
function build_graphene_hamiltonian_sparse(; radius::Real=20.0, star_amp::Real=0.2)
    h = LatticePresets.honeycomb() |>
        hopping(1) |>
        supercell(region = r -> norm(r) < radius * (1 + star_amp * cos(5 * atan(r[2], r[1]))))
    Hs = sparse(h(()))
    N  = size(Hs, 1)
    @assert size(Hs, 1) == size(Hs, 2)  "Hs no es cuadrada"
    return h, Hs, N
end

identity_permutation(N::Int) = collect(1:N)
apply_site_permutation(Hs::SparseMatrixCSC, perm::Vector{Int}) = Hs[perm, perm]

# =============================================================================
# SECCIÓN 2 — HAMILTONIANO TORUS (sin contorno; grafeno periódico)
# =============================================================================

"""
    build_honeycomb_torus_hamiltonian_sparse(Lx, Ly; t=1.0)

Red honeycomb con condiciones de contorno periódicas en x e y.
N = 2*Lx*Ly.
Vecinos NN:  A(x,y)–B(x,y),  A(x,y)–B(x-1,y),  A(x,y)–B(x,y-1).
"""
@inline _mod1(i::Int, L::Int) = mod1(i, L)

@inline function _honeycomb_idx(x::Int, y::Int, sub::Int, Lx::Int, Ly::Int)
    cell = (x - 1) + Lx * (y - 1)
    return 2 * cell + sub          # sub ∈ {1,2}
end

function build_honeycomb_torus_hamiltonian_sparse(Lx::Int, Ly::Int; t::Number=1.0)
    τ   = float(t)
    T   = promote_type(typeof(τ), typeof(conj(τ)))
    N   = 2 * Lx * Ly
    rows, cols, vals = Int[], Int[], T[]

    function add_hop!(i, j, amp)
        a = convert(T, amp)
        push!(rows, i); push!(cols, j); push!(vals, a)
        push!(rows, j); push!(cols, i); push!(vals, conj(a))
    end

    for x in 1:Lx, y in 1:Ly
        a  = _honeycomb_idx(x, y, 1, Lx, Ly)
        b0 = _honeycomb_idx(x, y, 2, Lx, Ly)
        b1 = _honeycomb_idx(_mod1(x-1,Lx), y,           2, Lx, Ly)
        b2 = _honeycomb_idx(x,             _mod1(y-1,Ly), 2, Lx, Ly)
        add_hop!(a, b0, -τ)
        add_hop!(a, b1, -τ)
        add_hop!(a, b2, -τ)
    end

    H = sparse(rows, cols, vals, N, N)
    dropzeros!(H)
    return H
end

# =============================================================================
# SECCIÓN 3 — CODIFICACIÓN QUANTICS ij (dim local = 4)
#
# μ = pair_to_localindex(br, bc) ∈ {1,2,3,4}
# donde br, bc ∈ {0,1} son los bits de la fila y columna en el nivel ℓ.
#
# La función F(μ_1,...,μ_R) = H[r,c] donde r,c se decodifican bit a bit.
# Esto es EXACTAMENTE el uso low-level de TCI.jl / Tensor4All:
#   - localdims = fill(4, R)
#   - F :: Vector{Int} -> T
#   - crossinterpolate2(T, F, localdims, pivots; ...)
# =============================================================================

@inline pair_to_localindex(br::Int, bc::Int) = 1 + br + 2*bc

@inline function localindex_to_pair(mu::Int)
    y  = mu - 1
    br = y & 0x1
    bc = (y >> 1) & 0x1
    return Int(br), Int(bc)
end

function index_to_bits(idx::Int, R::Int)
    x    = idx - 1
    bits = Vector{Int}(undef, R)
    @inbounds for ell in 1:R
        bits[ell] = (x >> (R - ell)) & 0x1
    end
    return bits
end

function encode_pair(r::Int, c::Int, R::Int)
    rb = index_to_bits(r, R)
    cb = index_to_bits(c, R)
    mu = Vector{Int}(undef, R)
    @inbounds for ell in 1:R
        mu[ell] = pair_to_localindex(rb[ell], cb[ell])
    end
    return mu
end

function decode_mu_ij(mu::AbstractVector{<:Integer})
    r = 0; c = 0
    @inbounds for x in mu
        br, bc = localindex_to_pair(x)
        r = (r << 1) | br
        c = (c << 1) | bc
    end
    return Int(r)+1, Int(c)+1
end

"""
    make_H_oracle(Hperm)

Construye el oráculo F(μ) = Hperm[r,c] para TCI.
Devuelve (F, R, M, N, T).
"""
function make_H_oracle(Hperm::SparseMatrixCSC)
    N = size(Hperm, 1)
    T = eltype(Hperm)
    R = max(1, ceil(Int, log2(max(N, 1))))
    M = 1 << R

    F = function(mu::AbstractVector{<:Integer})
        r, c = decode_mu_ij(mu)
        (r > N || c > N) && return zero(T)
        return Hperm[r, c]
    end

    return F, R, M, N, T
end

# =============================================================================
# SECCIÓN 4 — CODIFICACIÓN QUANTICS ijkl (dim local = 16)
#
# μ = quadbits_to_localindex(bi, bj, bk, bl) ∈ {1,...,16}
# donde (bi, bj) encodifican la fila y (bk, bl) la columna, en 2D.
# La matriz H_{IJ} se ve como tensor H_{i,j,k,l} con I=(i,j), J=(k,l).
# side_pad = 2^R,  N ≤ side_pad^2.
#
# DIFERENCIAS con ij:
#   - localdims = fill(16, R) con R más pequeño (log2(sqrt(N)))
#   - Mayor entrelazamiento espacial entre i,j,k,l en el mismo nivel
#   - En la práctica suele dar bond dims distintos a ij
# =============================================================================

@inline pair_to_global(i::Int, j::Int, side::Int) = 1 + (i-1) + side*(j-1)

@inline function global_to_pair(idx::Int, side::Int)
    x = idx - 1
    return (x % side) + 1, (x ÷ side) + 1
end

@inline quadbits_to_localindex(bi,bj,bk,bl) = 1 + bi + 2*bj + 4*bk + 8*bl

@inline function localindex_to_quadbits(mu::Int)
    y  = mu - 1
    bi = y & 0x1;        bj = (y >> 1) & 0x1
    bk = (y >> 2) & 0x1; bl = (y >> 3) & 0x1
    return Int(bi), Int(bj), Int(bk), Int(bl)
end

function encode_quad(i::Int, j::Int, k::Int, l::Int, R::Int)
    ib = index_to_bits(i, R); jb = index_to_bits(j, R)
    kb = index_to_bits(k, R); lb = index_to_bits(l, R)
    mu = Vector{Int}(undef, R)
    @inbounds for ell in 1:R
        mu[ell] = quadbits_to_localindex(ib[ell], jb[ell], kb[ell], lb[ell])
    end
    return mu
end

function decode_mu_ijkl(mu::AbstractVector{<:Integer})
    i=0; j=0; k=0; l=0
    @inbounds for x in mu
        bi,bj,bk,bl = localindex_to_quadbits(x)
        i=(i<<1)|bi; j=(j<<1)|bj; k=(k<<1)|bk; l=(l<<1)|bl
    end
    return Int(i)+1, Int(j)+1, Int(k)+1, Int(l)+1
end

function encode_global_indices(I::Int, J::Int, R::Int, side_pad::Int)
    i, j = global_to_pair(I, side_pad)
    k, l = global_to_pair(J, side_pad)
    return encode_quad(i, j, k, l, R)
end

function make_H_ijkl_oracle(Hperm::SparseMatrixCSC)
    N          = size(Hperm, 1)
    T          = eltype(Hperm)
    side_valid = ceil(Int, sqrt(N))
    R          = max(1, ceil(Int, log2(side_valid)))
    side_pad   = 1 << R
    Mglobal    = side_pad^2

    F = function(mu::AbstractVector{<:Integer})
        i, j, k, l = decode_mu_ijkl(mu)
        I = pair_to_global(i, j, side_pad)
        J = pair_to_global(k, l, side_pad)
        (I > N || J > N) && return zero(T)
        return Hperm[I, J]
    end

    return F, R, side_valid, side_pad, Mglobal, N, T
end

# =============================================================================
# SECCIÓN 5 — PIVOTS INICIALES
# =============================================================================

# --- ij ---

function _build_pivots_ij(Hperm, R; mode, nsample, extra, seed)
    I, J, V = findnz(Hperm)
    rng     = MersenneTwister(seed)
    seen    = Set{Any}()
    pivots  = Vector{Vector{Int}}()

    function _add!(i, j)
        p = encode_pair(i, j, R)
        tp = Tuple(p)
        tp in seen && return
        push!(seen, tp); push!(pivots, p)
    end

    _add!(1, 1)

    if mode === :all_nonzeros
        for t in eachindex(I); _add!(I[t], J[t]); end

    elseif mode === :random_nonzeros
        take = min(nsample, length(V))
        for t in randperm(rng, length(V))[1:take]; _add!(I[t], J[t]); end

    elseif mode === :rowcover
        rows_done = Dict{Int,Bool}(); cols_done = Dict{Int,Bool}()
        for t in eachindex(I)
            !haskey(rows_done, I[t]) && (_add!(I[t], J[t]); rows_done[I[t]]=true)
            !haskey(cols_done, J[t]) && (_add!(I[t], J[t]); cols_done[J[t]]=true)
        end
        take = min(extra, length(I))
        for t in randperm(rng, length(I))[1:take]; _add!(I[t], J[t]); end
    else
        error("pivot_mode desconocido: $mode")
    end
    return pivots
end

function choose_initial_pivots_ij(Hperm, R;
        pivot_mode=:rowcover, nsample=400, extra=300, seed=DEFAULT_SEED)
    return _build_pivots_ij(Hperm, R; mode=pivot_mode, nsample=nsample, extra=extra, seed=seed)
end

# --- ijkl ---

function _build_pivots_ijkl(Hperm, R, side_pad; mode, nsample, extra, seed)
    I, J, V = findnz(Hperm)
    rng     = MersenneTwister(seed)
    seen    = Set{Any}()
    pivots  = Vector{Vector{Int}}()

    function _add!(Ig, Jg)
        p = encode_global_indices(Ig, Jg, R, side_pad)
        tp = Tuple(p)
        tp in seen && return
        push!(seen, tp); push!(pivots, p)
    end

    _add!(1, 1)

    if mode === :all_nonzeros
        for t in eachindex(I); _add!(I[t], J[t]); end

    elseif mode === :random_nonzeros
        take = min(nsample, length(V))
        for t in randperm(rng, length(V))[1:take]; _add!(I[t], J[t]); end

    elseif mode === :rowcover
        rows_done = Dict{Int,Bool}(); cols_done = Dict{Int,Bool}()
        for t in eachindex(I)
            !haskey(rows_done, I[t]) && (_add!(I[t], J[t]); rows_done[I[t]]=true)
            !haskey(cols_done, J[t]) && (_add!(I[t], J[t]); cols_done[J[t]]=true)
        end
        take = min(extra, length(I))
        for t in randperm(rng, length(I))[1:take]; _add!(I[t], J[t]); end
    else
        error("pivot_mode desconocido: $mode")
    end
    return pivots
end

function choose_initial_pivots_ijkl(Hperm, R, side_pad;
        pivot_mode=:rowcover, nsample=400, extra=300, seed=DEFAULT_SEED)
    return _build_pivots_ijkl(Hperm, R, side_pad;
        mode=pivot_mode, nsample=nsample, extra=extra, seed=seed)
end

# =============================================================================
# SECCIÓN 6 — WRAPPERS TCI (robustos ante cambios de API)
# =============================================================================

function run_crossinterpolate2(::Type{T}, F, localdims, pivots;
        tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM, maxiter=DEFAULT_MAXITER,
        pivotsearch=:full) where {T}
    last_err = nothing
    kw_list = [
        (; tolerance, maxbonddim, maxiter, pivotsearch, normalizeerror=true, ncheckhistory=3, nsearchglobalpivot=0),
        (; tolerance, maxbonddim, maxiter, pivotsearch, normalizeerror=true, ncheckhistory=3),
        (; tolerance, maxbonddim, maxiter, pivotsearch, normalizeerror=true),
        (; tolerance, maxbonddim, maxiter, pivotsearch),
        (; tolerance, maxbonddim, maxiter),
        (; tolerance),
    ]
    for kw in kw_list
        try
            return TCI.crossinterpolate2(T, F, localdims, pivots; kw...)
        catch err
            last_err = err
        end
    end
    error("crossinterpolate2 falló. Último error: $last_err")
end

function canonicalize_if_possible!(tci, F; tolerance=DEFAULT_TOL)
    isdefined(TCI, :makecanonical!) || return tci
    for kw in ((; tolerance), (;))
        try; TCI.makecanonical!(tci, F; kw...); return tci; catch; end
    end
    try; TCI.makecanonical!(tci); return tci; catch; end
    return tci
end

function to_tensortrain(obj)
    isdefined(TCI, :TensorTrain) || return nothing
    try; return TCI.TensorTrain(obj);                catch; end
    try; return convert(TCI.TensorTrain, obj);        catch; end
    return nothing
end

function recompress_tt!(tt; method=:CI, tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM)
    (tt === nothing || !isdefined(TCI, :compress!)) && return tt
    for kw in ((; tolerance, maxbonddim), (; tolerance), (;))
        try; TCI.compress!(tt, method; kw...); return tt; catch; end
    end
    return tt
end

safe_linkdims(obj)   = TCI.linkdims(obj)
safe_evaluate(obj, mu) = TCI.evaluate(obj, mu)
safe_sitetensors(obj)  = collect(TCI.sitetensors(obj))

# =============================================================================
# SECCIÓN 7 — EXTRACCIÓN DE CORES MPO
# =============================================================================

struct CoreLayout
    perm::NTuple{3,Int}
    leftdim::Int
    physdim::Int
    rightdim::Int
end

function _candidate_layouts(core; physdim::Int)
    ndims(core) == 3 || error("Espero core 3D, size=$(size(core))")
    sz  = size(core)
    out = CoreLayout[]
    for p in 1:3
        sz[p] == physdim || continue
        rem = [a for a in 1:3 if a != p]
        push!(out, CoreLayout((rem[1],p,rem[2]), sz[rem[1]], sz[p], sz[rem[2]]))
        push!(out, CoreLayout((rem[2],p,rem[1]), sz[rem[2]], sz[p], sz[rem[1]]))
    end
    isempty(out) && error("No encontré eje físico=$physdim en core size=$sz")
    return out
end

function _all_chain_layouts(rawcores; physdim)
    L    = length(rawcores)
    cand = [_candidate_layouts(rawcores[ell]; physdim=physdim) for ell in 1:L]
    solutions = Vector{Vector{CoreLayout}}()
    current   = Vector{CoreLayout}(undef, L)

    function dfs(ell, prev_right)
        if ell > L
            prev_right == 1 && push!(solutions, copy(current))
            return
        end
        for lay in cand[ell]
            (ell == 1 ? lay.leftdim == 1 : lay.leftdim == prev_right) || continue
            current[ell] = lay
            dfs(ell+1, lay.rightdim)
        end
    end
    dfs(1, 1)
    isempty(solutions) && error("No encontré orientación cadena-consistente")
    return solutions
end

function _build_cores3(rawcores, layouts; physdim)
    T = eltype(rawcores[1])
    [let c = Array(PermutedDimsArray(rawcores[ell], layouts[ell].perm))
        size(c,2) == physdim || error("core $ell: physdim=$(size(c,2)) ≠ $physdim")
        c
    end for ell in eachindex(rawcores)]
end

function _eval_cores3(cores3, mu)
    L   = length(cores3)
    acc = reshape(Array(@view cores3[1][1, mu[1], :]), 1, :)
    for ell in 2:L-1
        acc = acc * Array(@view cores3[ell][:, mu[ell], :])
    end
    return (acc * reshape(Array(@view cores3[L][:, mu[L], 1]), :, 1))[1,1]
end

function _infer_layouts(rawcores, obj, test_mus; physdim)
    sols = _all_chain_layouts(rawcores; physdim=physdim)
    length(sols) == 1 && return sols[1], 0.0

    best_err = Inf; best_lay = nothing
    for layouts in sols
        cores3 = _build_cores3(rawcores, layouts; physdim=physdim)
        maxerr = 0.0
        for mu in test_mus
            ref = safe_evaluate(obj, mu)
            val = _eval_cores3(cores3, mu)
            maxerr = max(maxerr, abs(ref - val))
            maxerr > best_err && break
        end
        if maxerr < best_err
            best_err = maxerr; best_lay = layouts
        end
    end
    best_lay === nothing && error("No pude desambiguar la orientación de los cores")
    return best_lay, best_err
end

# ij: cores 4D  (bondL, rowbit, colbit, bondR)
function extract_W_ij(carrier, test_mus)
    rawcores = safe_sitetensors(carrier)
    layouts, layout_err = _infer_layouts(rawcores, carrier, test_mus; physdim=4)
    cores3 = _build_cores3(rawcores, layouts; physdim=4)
    T = eltype(cores3[1])
    W = Vector{Array{T,4}}(undef, length(cores3))
    for ell in eachindex(cores3)
        Lb, pd, Rb = size(cores3[ell])
        W[ell] = reshape(Array(cores3[ell]), Lb, 2, 2, Rb)
    end
    return W, cores3, layout_err
end

# ijkl: cores 6D  (bondL, bi, bj, bk, bl, bondR)
function extract_W_ijkl(carrier, test_mus)
    rawcores = safe_sitetensors(carrier)
    layouts, layout_err = _infer_layouts(rawcores, carrier, test_mus; physdim=16)
    cores3 = _build_cores3(rawcores, layouts; physdim=16)
    T  = eltype(cores3[1])
    W  = Vector{Array{T,6}}(undef, length(cores3))
    for ell in eachindex(cores3)
        Lb,pd,Rb = size(cores3[ell])
        A = zeros(T, Lb, 2, 2, 2, 2, Rb)
        for a in 1:Lb, mu in 1:16, b in 1:Rb
            bi,bj,bk,bl = localindex_to_quadbits(mu)
            A[a,bi+1,bj+1,bk+1,bl+1,b] = cores3[ell][a,mu,b]
        end
        W[ell] = A
    end
    return W, cores3, layout_err
end






# =============================================================================
# SECCIÓN 8 — EVALUACIÓN DEL MPO
# =============================================================================

"""Evalúa W (formato ij) en (r, c)."""
function evaluate_mpo_ij(W, r::Int, c::Int, R::Int)
    rb = index_to_bits(r, R)
    cb = index_to_bits(c, R)
    acc = reshape(Array(@view W[1][1, rb[1]+1, cb[1]+1, :]), 1, :)
    for ell in 2:R-1
        acc = acc * Array(@view W[ell][:, rb[ell]+1, cb[ell]+1, :])
    end
    lastcol = reshape(Array(@view W[R][:, rb[R]+1, cb[R]+1, 1]), :, 1)
    return (acc * lastcol)[1,1]
end

"""Evalúa W (formato ijkl) en (I, J) usando side_pad."""
function evaluate_mpo_ijkl(W, I::Int, J::Int, R::Int, side_pad::Int)
    i,j = global_to_pair(I, side_pad)
    k,l = global_to_pair(J, side_pad)
    ib=index_to_bits(i,R); jb=index_to_bits(j,R)
    kb=index_to_bits(k,R); lb=index_to_bits(l,R)
    acc = reshape(Array(@view W[1][1, ib[1]+1,jb[1]+1,kb[1]+1,lb[1]+1, :]), 1, :)
    for ell in 2:R-1
        acc = acc * Array(@view W[ell][:, ib[ell]+1,jb[ell]+1,kb[ell]+1,lb[ell]+1, :])
    end
    lastcol = reshape(Array(@view W[R][:, ib[R]+1,jb[R]+1,kb[R]+1,lb[R]+1,1]), :, 1)
    return (acc * lastcol)[1,1]
end

# =============================================================================
# SECCIÓN 9 — VALIDACIÓN Y MÉTRICAS
# =============================================================================

function sample_nz_and_zero(Hperm; nsamples_nz=4000, nsamples_zero=4000, seed=DEFAULT_SEED)
    I, J, V = findnz(Hperm)
    rng  = MersenneTwister(seed)
    N    = size(Hperm, 1)
    take = min(nsamples_nz, length(V))
    nz   = [(I[t], J[t], V[t]) for t in randperm(rng, length(V))[1:take]]
    zeros_ = Tuple{Int,Int}[]
    rng2 = MersenneTwister(seed+1)
    while length(zeros_) < nsamples_zero
        i = rand(rng2, 1:N); j = rand(rng2, 1:N)
        Hperm[i,j] == 0 && push!(zeros_, (i,j))
    end
    return nz, zeros_
end

function sampled_validation(Hperm, evalfun;
        nsamples_nz=4000, nsamples_zero=4000, nsamples_herm=4000, seed=DEFAULT_SEED)
    nz, zeros_ = sample_nz_and_zero(Hperm; nsamples_nz, nsamples_zero, seed)
    rng  = MersenneTwister(seed+2)
    N    = size(Hperm, 1)

    max_nz=0.0; rms_nz=0.0
    for (i,j,v) in nz
        e = abs(evalfun(i,j) - v)
        max_nz = max(max_nz, e); rms_nz += e^2
    end
    rms_nz = sqrt(rms_nz / max(length(nz),1))

    max_z=0.0; rms_z=0.0
    for (i,j) in zeros_
        e = abs(evalfun(i,j))
        max_z = max(max_z, e); rms_z += e^2
    end
    rms_z = sqrt(rms_z / max(length(zeros_),1))

    max_herm=0.0
    for _ in 1:nsamples_herm
        i = rand(rng, 1:N); j = rand(rng, 1:N)
        max_herm = max(max_herm, abs(evalfun(i,j) - conj(evalfun(j,i))))
    end

    return (max_abs_err_nonzero = max_nz, rms_err_nonzero = rms_nz,
            max_abs_on_sampled_zeros = max_z, rms_on_sampled_zeros = rms_z,
            max_hermiticity_violation = max_herm)
end

function validate_all_nonzeros(Hperm, evalfun; zero_samples=5000, seed=DEFAULT_SEED)
    rng     = MersenneTwister(seed)
    I,J,V   = findnz(Hperm)
    N       = size(Hperm,1)
    errs_nz = Float64[]
    bad     = Any[]

    for t in eachindex(V)
        err = abs(evalfun(I[t], J[t]) - V[t])
        push!(errs_nz, err)
        err > 1e-10 && push!(bad, (i=I[t], j=J[t], exact=V[t], approx=evalfun(I[t],J[t]), err=err))
    end

    errs_z = Float64[]; cnt=0
    while cnt < zero_samples
        i=rand(rng,1:N); j=rand(rng,1:N)
        if Hperm[i,j]==0; push!(errs_z, abs(evalfun(i,j))); cnt+=1; end
    end

    return (max_abs_err_nonzero = maximum(errs_nz),
            rms_err_nonzero     = sqrt(mean(abs2, errs_nz)),
            max_abs_on_zeros    = isempty(errs_z) ? 0.0 : maximum(errs_z),
            n_bad = length(bad), bad = bad)
end

function frobenius_norm2_mpo_ij(W)
    T   = eltype(W[1])
    env = ones(promote_type(Float64,T), 1, 1)
    for ell in eachindex(W)
        cL,_,_,cR = size(W[ell])
        env_next = zeros(promote_type(Float64,T), cR, cR)
        @inbounds for a1 in 1:cL, b1 in 1:cL
            e = env[a1,b1]; e==0 && continue
            for a2 in 1:cR, b2 in 1:cR
                s = zero(promote_type(Float64,T))
                for jr in 1:2, kc in 1:2
                    s += conj(W[ell][a1,jr,kc,a2]) * W[ell][b1,jr,kc,b2]
                end
                env_next[a2,b2] += e * s
            end
        end
        env = env_next
    end
    return real(env[1,1])
end

linkdims_of(out) = out.tt === nothing ? safe_linkdims(out.tci) : safe_linkdims(out.tt)

# =============================================================================
# SECCIÓN 10 — LOGGING
# =============================================================================

_log_step(letter, title) = println("\n▣ PASO $letter — $title")
_log_kv(k, v)            = println("   · $k = $v")
_log_info(msg)           = println("   · $msg")
function _log_nt(title, nt)
    println("   · $title")
    for (k,v) in pairs(nt); println("      - $k = $v"); end
end

# =============================================================================
# SECCIÓN 11 — PIPELINE IJ COMPLETO
# =============================================================================

"""
    build_graphene_H_mpo_ij(; radius, star_amp, ...)

Pipeline QTCI completo para el flake de grafeno, codificación ij (dim local 4).
Devuelve named tuple con todos los resultados y validaciones.
"""
function build_graphene_H_mpo_ij(;
    radius::Real       = 20.0,
    star_amp::Real     = 0.2,
    perm               = nothing,
    tolerance          = DEFAULT_TOL,
    maxbonddim::Int    = DEFAULT_MAXBONDDIM,
    maxiter::Int       = DEFAULT_MAXITER,
    seed::Int          = DEFAULT_SEED,
    pivot_mode::Symbol = :rowcover,
    nsample_pivots::Int = 400,
    extra_pivots::Int   = 300,
    do_recompress::Bool = true,
    recompress_method::Symbol = :CI,
    validate_samples::Bool = true,
    validate_all_nz::Bool  = false,
)
    _log_step("A","construir H sparse (flake con contorno)")
    h, Hs, N = build_graphene_hamiltonian_sparse(; radius, star_amp)
    _log_kv("N", N); _log_kv("nnz(H)", nnz(Hs))

    _log_step("B","permutación de sitios")
    perm === nothing && (perm = identity_permutation(N); _log_info("Usando permutación identidad"))
    Hperm = apply_site_permutation(Hs, perm)

    _log_step("C","oráculo quantics F(μ)")
    F, R, M, Nvalid, T = make_H_oracle(Hperm)
    localdims = fill(4, R)
    _log_kv("R", R); _log_kv("localdims", localdims)

    _log_step("D","pivots iniciales ($pivot_mode)")
    pivots = choose_initial_pivots_ij(Hperm, R;
        pivot_mode=pivot_mode, nsample=nsample_pivots, extra=extra_pivots, seed)
    _log_kv("n_pivots", length(pivots))

    _log_step("E","TCI / crossinterpolate2")
    tci, ranks_hist, err_hist = run_crossinterpolate2(T, F, localdims, pivots;
        tolerance, maxbonddim, maxiter)
    _log_kv("linkdims tras TCI", safe_linkdims(tci))

    _log_step("F","canonicalización")
    canonicalize_if_possible!(tci, F; tolerance)

    tci_val = nothing
    if validate_samples
        _log_step("G","validación TCI")
        tci_eval = (i,j) -> safe_evaluate(tci, encode_pair(i,j,R))
        tci_val  = sampled_validation(Hperm, tci_eval; seed=seed+100)
        _log_nt("TCI validation", tci_val)
    end

    _log_step("H","conversión a TensorTrain")
    tt = to_tensortrain(tci)
    if tt !== nothing
        _log_info("TensorTrain OK; linkdims = $(safe_linkdims(tt))")
        if do_recompress
            _log_step("I","recompresión")
            recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim)
            _log_kv("linkdims tras recompresión", safe_linkdims(tt))
        end
    else
        _log_info("TensorTrain no disponible; se usa TCI como carrier")
    end

    carrier = tt === nothing ? tci : tt

    _log_step("J","extracción de cores W")
    I_nz, J_nz, _ = findnz(Hperm)
    rng_lay = MersenneTwister(seed+149)
    n_lay   = min(length(I_nz), 512)
    sel_lay = randperm(rng_lay, length(I_nz))[1:n_lay]
    test_mus = [encode_pair(I_nz[t], J_nz[t], R) for t in sel_lay]
    W, cores3, layout_err = extract_W_ij(carrier, test_mus)
    _log_kv("size.(W)", size.(W))
    _log_kv("layout_error", layout_err)

    _log_step("K","validación W MPO")
    W_eval = (i,j) -> evaluate_mpo_ij(W, i, j, R)
    W_val  = sampled_validation(Hperm, W_eval; seed=seed+200)
    _log_nt("W validation", W_val)

    fro_norm = sqrt(max(frobenius_norm2_mpo_ij(W), 0.0))
    _log_kv("‖W‖_F", fro_norm)

    all_nz_val = nothing
    if validate_all_nz
        _log_step("L","validación en todos los no nulos")
        all_nz_val = validate_all_nonzeros(Hperm, W_eval; seed=seed+400)
        _log_nt("all_nonzeros_validation", all_nz_val)
    end

    return (h=h, Hs=Hs, Hperm=Hperm, perm=perm, F=F, R=R, M=M, N=N,
            localdims=localdims, pivots=pivots, tci=tci, tt=tt, carrier=carrier,
            ranks_hist=ranks_hist, err_hist=err_hist, cores3=cores3, W=W,
            layout_err=layout_err, tci_val=tci_val, W_val=W_val,
            all_nz_val=all_nz_val, fro_norm=fro_norm)
end

# Wrapper para torus con pipeline ij
function build_torus_H_mpo_ij(Lx::Int, Ly::Int;
        t::Real=1.0, kwargs...)
    H = build_honeycomb_torus_hamiltonian_sparse(Lx, Ly; t)
    perm = identity_permutation(size(H,1))
    Hperm = H
    R = max(1, ceil(Int, log2(max(size(H,1), 1))))
    M = 1 << R
    F, R2, M2, N, T = make_H_oracle(Hperm)

    # Reutilizamos build_graphene_H_mpo_ij pero con H prebuilded
    # (pasando perm explícito evitamos rebuild de Quantica)
    return _run_tci_ij_pipeline(H, Hperm, perm; kwargs...)
end

# Helper interno compartido
function _run_tci_ij_pipeline(Hs, Hperm, perm;
        tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM, maxiter=DEFAULT_MAXITER,
        seed=DEFAULT_SEED, pivot_mode=:rowcover, nsample_pivots=400, extra_pivots=300,
        do_recompress=true, recompress_method=:CI,
        validate_samples=true, validate_all_nz=false)

    N = size(Hperm, 1)
    F, R, M, _, T = make_H_oracle(Hperm)
    localdims = fill(4, R)

    pivots = choose_initial_pivots_ij(Hperm, R;
        pivot_mode, nsample=nsample_pivots, extra=extra_pivots, seed)

    tci, ranks_hist, err_hist = run_crossinterpolate2(T, F, localdims, pivots;
        tolerance, maxbonddim, maxiter)
    canonicalize_if_possible!(tci, F; tolerance)

    tci_val = validate_samples ? sampled_validation(Hperm,
        (i,j)->safe_evaluate(tci,encode_pair(i,j,R)); seed=seed+100) : nothing

    tt = to_tensortrain(tci)
    if tt !== nothing && do_recompress
        recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim)
    end
    carrier = tt === nothing ? tci : tt

    I_nz,J_nz,_ = findnz(Hperm)
    rng_lay = MersenneTwister(seed+149)
    n_lay   = min(length(I_nz), 512)
    sel_lay = randperm(rng_lay, length(I_nz))[1:n_lay]
    test_mus = [encode_pair(I_nz[t], J_nz[t], R) for t in sel_lay]
    W, cores3, layout_err = extract_W_ij(carrier, test_mus)

    W_eval = (i,j) -> evaluate_mpo_ij(W, i, j, R)
    W_val  = sampled_validation(Hperm, W_eval; seed=seed+200)
    fro_norm = sqrt(max(frobenius_norm2_mpo_ij(W), 0.0))

    all_nz_val = validate_all_nz ?
        validate_all_nonzeros(Hperm, W_eval; seed=seed+400) : nothing

    return (Hs=Hs, Hperm=Hperm, perm=perm, F=F, R=R, M=M, N=N,
            localdims=localdims, pivots=pivots, tci=tci, tt=tt, carrier=carrier,
            ranks_hist=ranks_hist, err_hist=err_hist, cores3=cores3, W=W,
            layout_err=layout_err, tci_val=tci_val, W_val=W_val,
            all_nz_val=all_nz_val, fro_norm=fro_norm)
end

# Alias de compatibilidad con el nombre antiguo
build_graphene_H_mpo_clean(args...; kwargs...) = build_graphene_H_mpo_ij(args...; kwargs...)

# =============================================================================
# SECCIÓN 12 — PIPELINE IJKL COMPLETO
# =============================================================================

"""
    build_graphene_H_mpo_ijkl(; radius, star_amp, ...)

Pipeline QTCI para el flake, codificación ijkl (dim local 16).
"""
function build_graphene_H_mpo_ijkl(;
    radius::Real       = 20.0,
    star_amp::Real     = 0.2,
    perm               = nothing,
    tolerance          = DEFAULT_TOL,
    maxbonddim::Int    = DEFAULT_MAXBONDDIM,
    maxiter::Int       = DEFAULT_MAXITER,
    seed::Int          = DEFAULT_SEED,
    pivot_mode::Symbol = :rowcover,
    nsample_pivots::Int = 400,
    extra_pivots::Int   = 300,
    do_recompress::Bool = true,
    recompress_method::Symbol = :CI,
    validate_samples::Bool = true,
    validate_all_nz::Bool  = false,
    verbosity::Symbol      = :compact,
)
    verbosity != :quiet && println("▶ Construcción H sparse (flake ijkl)")
    h, Hs, N = build_graphene_hamiltonian_sparse(; radius, star_amp)

    perm === nothing && (perm = identity_permutation(N))
    Hperm = apply_site_permutation(Hs, perm)

    F, R, side_valid, side_pad, Mglobal, Nvalid, T = make_H_ijkl_oracle(Hperm)
    localdims = fill(16, R)

    verbosity != :quiet && println("   N=$N | R=$R | side_pad=$side_pad | localdims=16^$R")

    pivots = choose_initial_pivots_ijkl(Hperm, R, side_pad;
        pivot_mode, nsample=nsample_pivots, extra=extra_pivots, seed)

    tci, ranks_hist, err_hist = run_crossinterpolate2(T, F, localdims, pivots;
        tolerance, maxbonddim, maxiter)
    canonicalize_if_possible!(tci, F; tolerance)

    lds_tci = safe_linkdims(tci)
    verbosity != :quiet && println("   TCI χmax = $(maximum(lds_tci))")

    tci_val = validate_samples ? sampled_validation(Hperm,
        (I,J)->safe_evaluate(tci, encode_global_indices(I,J,R,side_pad)); seed=seed+100) : nothing

    tt = to_tensortrain(tci)
    if tt !== nothing && do_recompress
        recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim)
    end
    carrier = tt === nothing ? tci : tt

    I_nz,J_nz,_ = findnz(Hperm)
    rng_lay = MersenneTwister(seed+149)
    n_lay   = min(length(I_nz), 64)
    sel_lay = randperm(rng_lay, length(I_nz))[1:n_lay]
    test_mus = [encode_global_indices(I_nz[t], J_nz[t], R, side_pad) for t in sel_lay]
    W, cores3, layout_err = extract_W_ijkl(carrier, test_mus)

    W_eval = (I,J) -> evaluate_mpo_ijkl(W, I, J, R, side_pad)
    W_val  = validate_samples ? sampled_validation(Hperm, W_eval; seed=seed+200) : nothing

    all_nz_val = validate_all_nz ?
        validate_all_nonzeros(Hperm, W_eval; seed=seed+400) : nothing

    return (h=h, Hs=Hs, Hperm=Hperm, perm=perm, F=F, R=R, side_valid=side_valid,
            side_pad=side_pad, Mglobal=Mglobal, N=N, localdims=localdims, pivots=pivots,
            tci=tci, tt=tt, carrier=carrier, ranks_hist=ranks_hist, err_hist=err_hist,
            cores3=cores3, W=W, layout_err=layout_err, tci_val=tci_val, W_val=W_val,
            all_nz_val=all_nz_val)
end

# Alias de compatibilidad
build_graphene_H_ijkl_mpo_clean(args...; kwargs...) = build_graphene_H_mpo_ijkl(args...; kwargs...)

# =============================================================================
# SECCIÓN 13 — MPO EXACTO (sparse → autómata DAG → MPO)
#
# MÉTODO: Cada no nulo H[i,j] se codifica como una palabra μ_1...μ_R ∈ {1..4}^R.
# Se construye un DAG de prefijos compartidos (sufijo-canonical):
#   - Nivel ℓ < R: estados = clases de equivalencia de subpalabras a la derecha
#   - Nivel R:     estados finales con valores escalares
# El DAG se traduce directamente a tensores MPO.
# Resultado: EXACTO (error = redondeo float), sin TCI.
#
# DIFERENCIA con AutoMPO (ITensors):
#   - AutoMPO construye un MPO many-body para c†_i c_j
#   - El MPO exacto aquí opera sobre el ESPACIO DE HILBERT CUÁNTICO DE LA MATRIZ,
#     es decir, indexado por los bits cuánticos (rowbit, colbit), NO por fermiones.
#   - Son dos representaciones distintas del mismo Hamiltoniano.
# =============================================================================

function _collect_sparse_words(Hperm)
    I,J,V = findnz(Hperm)
    N = size(Hperm,1)
    R = max(1, ceil(Int, log2(max(N,1))))
    T = eltype(Hperm)
    words = [encode_pair(I[t], J[t], R) for t in eachindex(V)]
    return words, collect(V), R
end

function _build_suffix_dag(words, vals::Vector{T}, R) where {T}
    R >= 1 || error("R debe ser ≥ 1")

    _prefix(w, d) = d == 0 ? () : Tuple(w[1:d])

    if R == 1
        fm = Dict{Int,T}()
        for (w,v) in zip(words, vals); fm[w[1]] = get(fm, w[1], zero(T)) + v; end
        return Any[[fm]]
    end

    level_map  = Vector{Any}(undef, R)
    level_desc = Vector{Any}(undef, R)

    # último nivel: tablas de valores
    tmp_last = Dict{Any, Dict{Int,T}}()
    for (w,v) in zip(words, vals)
        p = _prefix(w, R-1)
        d = get!(tmp_last, p, Dict{Int,T}())
        d[w[R]] = get(d, w[R], zero(T)) + v
    end
    sig2s = Dict{Any,Int}(); map_last = Dict{Any,Int}(); desc_last = Vector{Dict{Int,T}}()
    for (p,d) in tmp_last
        sig = Tuple(sort!(collect(d); by=first))
        sid = get(sig2s, sig, 0)
        if sid == 0; sid = length(desc_last)+1; sig2s[sig]=sid; push!(desc_last, Dict(collect(sig))); end
        map_last[p] = sid
    end
    level_map[R]  = map_last
    level_desc[R] = desc_last

    # niveles intermedios
    for depth in (R-2):-1:0
        tmp = Dict{Any, Dict{Int,Int}}()
        for (w,_) in zip(words, vals)
            p      = _prefix(w, depth)
            cp     = _prefix(w, depth+1)
            μ      = w[depth+1]
            cstate = level_map[depth+2][cp]
            d      = get!(tmp, p, Dict{Int,Int}())
            d[μ]   = cstate
        end
        sig2s2 = Dict{Any,Int}(); this_map = Dict{Any,Int}(); this_desc = Vector{Dict{Int,Int}}()
        for (p,d) in tmp
            sig = Tuple(sort!(collect(d); by=first))
            sid = get(sig2s2, sig, 0)
            if sid == 0; sid = length(this_desc)+1; sig2s2[sig]=sid; push!(this_desc, Dict(collect(sig))); end
            this_map[p] = sid
        end
        level_map[depth+1]  = this_map
        level_desc[depth+1] = this_desc
    end
    return level_desc
end

function _dag_to_mpo(Hperm::SparseMatrixCSC)
    words, vals, R = _collect_sparse_words(Hperm)
    T = eltype(Hperm)

    isempty(vals) && return [zeros(T,1,2,2,1) for _ in 1:R]

    level_desc = _build_suffix_dag(words, vals, R)

    if R == 1
        W = [zeros(T,1,2,2,1)]
        for (μ,v) in level_desc[1][1]
            br,bc = localindex_to_pair(μ)
            W[1][1,br+1,bc+1,1] += v
        end
        return W
    end

    W = Vector{Array{T,4}}(undef, R)

    for ℓ in 1:R-1
        dh = level_desc[ℓ]; dn = level_desc[ℓ+1]
        χL = length(dh); χR = length(dn)
        W[ℓ] = zeros(T, χL, 2, 2, χR)
        for a in 1:χL, (μ,b) in dh[a]
            br,bc = localindex_to_pair(μ)
            W[ℓ][a, br+1, bc+1, b] += one(T)
        end
    end

    dl = level_desc[R]; χL = length(dl)
    W[R] = zeros(T, χL, 2, 2, 1)
    for a in 1:χL, (μ,v) in dl[a]
        br,bc = localindex_to_pair(μ)
        W[R][a, br+1, bc+1, 1] += v
    end

    return W
end

"""
    build_sparse_H_mpo_exact(Hs; validate_samples, zero_samples, seed, verbose)

Construye el MPO quantics exacto de la matriz sparse Hs mediante autómata DAG
de palabras quantics (sin TCI). Resultado es exacto salvo redondeo float.

ATENCIÓN: Este MPO es una representación de la MATRIZ H como TT cuántico,
indexado por (rowbit, colbit). NO es un AutoMPO many-body de ITensors.
"""
function build_sparse_H_mpo_exact(Hs::SparseMatrixCSC;
        perm=nothing, validate_samples=true, zero_samples=5000,
        seed=DEFAULT_SEED, verbose=true)
    N = size(Hs,1)
    size(Hs,1)==size(Hs,2) || error("Hs debe ser cuadrada")

    perm === nothing && (perm = identity_permutation(N))
    Hperm = apply_site_permutation(Hs, perm)

    verbose && println("\n▣ MPO exacto sparse → quantics-MPO")
    verbose && _log_kv("N", N); verbose && _log_kv("nnz(Hperm)", nnz(Hperm))

    W   = _dag_to_mpo(Hperm)
    R   = length(W)
    lds = [size(W[ℓ],4) for ℓ in 1:R-1]
    χmax = isempty(lds) ? 1 : maximum(lds)
    storage = sum(length, W)

    verbose && _log_kv("R", R)
    verbose && _log_kv("linkdims", lds)
    verbose && _log_kv("χmax", χmax)
    verbose && _log_kv("storage", storage)

    W_eval = (i,j) -> evaluate_mpo_ij(W, i, j, R)

    sampled_val = nothing; all_nz_val = nothing
    if validate_samples
        sampled_val = sampled_validation(Hperm, W_eval; seed)
        all_nz_val  = validate_all_nonzeros(Hperm, W_eval; zero_samples, seed=seed+1)
        verbose && _log_nt("sampled_validation", sampled_val)
        verbose && _log_kv("n_bad", all_nz_val.n_bad)
        verbose && _log_kv("max_abs_err_nonzero", all_nz_val.max_abs_err_nonzero)
    end

    return (Hs=Hs, Hperm=Hperm, perm=perm, N=N, R=R, W=W,
            linkdims=lds, χmax=χmax, storage=storage,
            sampled_validation=sampled_val, all_nonzeros_validation=all_nz_val)
end

function build_graphene_flake_H_mpo_exact(;
        radius=20.0, star_amp=0.2, kwargs...)
    h, Hs, N = build_graphene_hamiltonian_sparse(; radius, star_amp)
    out = build_sparse_H_mpo_exact(Hs; kwargs...)
    return merge(out, (h=h, radius=radius, star_amp=star_amp))
end

function build_graphene_torus_H_mpo_exact(Lx, Ly; t=1.0, kwargs...)
    H = build_honeycomb_torus_hamiltonian_sparse(Lx, Ly; t)
    return build_sparse_H_mpo_exact(H; kwargs...)
end

function pretty_exact_summary(out)
    println("RESUMEN MPO EXACTO")
    println("  N        = $(out.N)")
    println("  R        = $(out.R)")
    println("  linkdims = $(out.linkdims)")
    println("  χmax     = $(out.χmax)")
    println("  storage  = $(out.storage)")
    println("  size.(W) = $(size.(out.W))")
    if out.all_nonzeros_validation !== nothing
        println("  n_bad    = $(out.all_nonzeros_validation.n_bad)")
        println("  max_err  = $(out.all_nonzeros_validation.max_abs_err_nonzero)")
    end
end

# =============================================================================
# SECCIÓN 14 — AutoMPO (ITensors many-body)
#
# NOTA CONCEPTUAL IMPORTANTE:
#   El AutoMPO opera en el espacio de Fock many-body de fermiones.
#   El operador resultante actúa sobre estados del tipo |n_1, n_2, ..., n_N⟩
#   y es el Hamiltoniano H = Σ_{ij} H_{ij} c†_i c_j.
#   Es DIFERENTE del MPO cuántico-quantics del sparse H:
#     - quantics MPO: representa la MATRIZ H_{ij} como TT cuántico
#     - AutoMPO: representa el OPERADOR many-body en el espacio de Fock
# =============================================================================

function build_graphene_autompo(Hs::SparseMatrixCSC; atol=1e-12)
    _HAVE_ITENSORS || error("ITensors no está cargado. Ejecuta 'using ITensors, ITensorMPS'.")

    N  = size(Hs, 1)
    os = OpSum()
    I, J, V = findnz(Hs)
    for t in eachindex(V)
        i, j, v = I[t], J[t], V[t]
        if i == j
            abs(imag(v)) > atol && @warn "Diagonal compleja" i v
            abs(real(v)) > atol && add!(os, real(v), "N", i)
        else
            abs(v) > atol && add!(os, v, "Cdag", i, "C", j)
        end
    end
    return os
end

"""
    build_graphene_flake_autompo(; radius, star_amp, splitblocks)

Construye el MPO many-body via AutoMPO (ITensors) para el flake.
Para N grande (>500), esto puede ser lento; usar con radio pequeño para demo.
"""
function build_graphene_flake_autompo(; radius=5.0, star_amp=0.2, splitblocks=true)
    _HAVE_ITENSORS || error("ITensors no está cargado.")
    h, Hs, N = build_graphene_hamiltonian_sparse(; radius, star_amp)
    println("N=$N | nnz=$(nnz(Hs))")
    sites = siteinds("Fermion", N)
    os    = build_graphene_autompo(Hs)
    mpo   = MPO(os, sites; splitblocks=splitblocks)
    return (h=h, Hs=Hs, N=N, sites=sites, os=os, mpo=mpo)
end

using LinearAlgebra
using SparseArrays
using Quantica

# ============================================================
# AutoMPO / ITensors (many-body fermiónico)
#
# Esta sección representa el operador many-body
#     H = Σ_{ij} Hs[i,j] c†_i c_j
# en el espacio de Fock fermiónico.
#
# OJO:
# - Esto NO es el MPO quantics de la matriz H_{ij}.
# - Aquí construimos el operador many-body con AutoMPO/OpSum.
# ============================================================

# ------------------------------------------------------------
# Carga opcional de ITensors
# ------------------------------------------------------------
const _HAVE_ITENSORS_AUTOMPO = try
    using ITensors, ITensorMPS
    true
catch err
    @warn "No se pudo cargar ITensors/ITensorMPS" exception=(err, catch_backtrace())
    false
end

function _check_itensors_autompo()
    _HAVE_ITENSORS_AUTOMPO || error(
        "ITensors/ITensorMPS no está cargado. " *
        "Instala/carga esos paquetes antes de usar esta sección."
    )
    return nothing
end

# ------------------------------------------------------------
# Hamiltoniano sparse del flake de grafeno
# ------------------------------------------------------------
"""
    build_graphene_hamiltonian_sparse(; radius=20.0, star_amp=0.2)

Construye el Hamiltoniano tight-binding sparse del flake de grafeno
con contorno estrellado.

Devuelve:
- h   :: objeto Quantica Hamiltonian
- Hs  :: SparseMatrixCSC
- N   :: número de sitios
"""
function build_graphene_hamiltonian_sparse(; radius::Real=20.0, star_amp::Real=0.2)
    h = LatticePresets.honeycomb() |>
        hopping(1) |>
        supercell(
            region = r -> norm(r) < radius * (1 + star_amp * cos(5 * atan(r[2], r[1])))
        )

    Hs = sparse(h(()))
    N  = size(Hs, 1)

    size(Hs, 1) == size(Hs, 2) || error("Hs no es cuadrada")
    return h, Hs, N
end

# ------------------------------------------------------------
# Hamiltoniano sparse del torus honeycomb periódico
# ------------------------------------------------------------
@inline _mod1_local(i::Int, L::Int) = mod1(i, L)

@inline function _honeycomb_idx(x::Int, y::Int, sub::Int, Lx::Int, Ly::Int)
    cell = (x - 1) + Lx * (y - 1)
    return 2 * cell + sub   # sub ∈ {1,2}
end

"""
    build_honeycomb_torus_hamiltonian_sparse(Lx, Ly; t=1.0)

Construye el Hamiltoniano sparse de una red honeycomb periódica (torus).

N = 2 * Lx * Ly
"""
function build_honeycomb_torus_hamiltonian_sparse(Lx::Int, Ly::Int; t::Number=1.0)
    τ = float(t)
    T = promote_type(typeof(τ), typeof(conj(τ)))

    N = 2 * Lx * Ly
    rows = Int[]
    cols = Int[]
    vals = T[]

    function add_hop!(i::Int, j::Int, amp)
        a = convert(T, amp)
        push!(rows, i); push!(cols, j); push!(vals, a)
        push!(rows, j); push!(cols, i); push!(vals, conj(a))
    end

    for x in 1:Lx, y in 1:Ly
        a  = _honeycomb_idx(x, y, 1, Lx, Ly)
        b0 = _honeycomb_idx(x, y, 2, Lx, Ly)
        b1 = _honeycomb_idx(_mod1_local(x - 1, Lx), y, 2, Lx, Ly)
        b2 = _honeycomb_idx(x, _mod1_local(y - 1, Ly), 2, Lx, Ly)

        add_hop!(a, b0, -τ)
        add_hop!(a, b1, -τ)
        add_hop!(a, b2, -τ)
    end

    H = sparse(rows, cols, vals, N, N)
    dropzeros!(H)
    return H
end

# ------------------------------------------------------------
# Sparse matrix -> OpSum
# ------------------------------------------------------------
"""
    build_graphene_opsum(Hs; atol=1e-12)

Convierte la matriz sparse Hs en un OpSum fermiónico:

    H = Σ_i Hii N_i + Σ_{i≠j} Hij c†_i c_j

Notas:
- Los términos diagonales se traducen como "N", i
- Los términos no diagonales como "Cdag", i, "C", j
- Si hay parte imaginaria en la diagonal por encima de atol, avisa
"""
function build_graphene_opsum(Hs::SparseMatrixCSC; atol::Real=1e-12)
    _check_itensors_autompo()

    size(Hs, 1) == size(Hs, 2) || error("Hs debe ser cuadrada")

    os = OpSum()
    I, J, V = findnz(Hs)

    for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]

        if i == j
            abs(imag(v)) > atol && @warn "Diagonal compleja detectada" i j v
            abs(real(v)) > atol && add!(os, real(v), "N", i)
        else
            abs(v) > atol && add!(os, v, "Cdag", i, "C", j)
        end
    end

    return os
end

# ============================================================
# Resumen del MPO fermiónico AutoMPO (many-body en Fock)
# ============================================================
using ITensors, ITensorMPS

function pretty_fermion_mpo_summary(out)
    mpo   = out.mpo
    sites = out.sites
    N     = length(mpo)
    lds   = linkdims(mpo)
    χmax  = isempty(lds) ? 1 : maximum(lds)

    println("RESUMEN AutoMPO / MPO fermiónico")
    println("  N sitios físicos (Fock) = ", out.N)
    println("  longitud cadena MPO     = ", N)
    println("  dim local por sitio     = ", unique(dim.(sites)))
    println("  linkdims                = ", lds)
    println("  χmax                    = ", χmax)
    println("  n legs por tensor       = ", [length(inds(mpo[n])) for n in 1:N])

    # Dimensiones de los tensores del MPO
    tens_dims = [dim.(collect(inds(mpo[n]))) for n in 1:N]
    println("  dims de cada tensor     = ", tens_dims)

    println("\nInterpretación:")
    println("  - Cada sitio del MPO es un sitio fermiónico local con d=2: |Emp>, |Occ>")
    println("  - Los tensores interiores tienen 4 legs: link izq, site bra, site ket, link der")
    println("  - En los bordes suele haber 3 legs porque un link tiene dimensión 1")
    println("  - Este MPO vive en Fock many-body, NO en quantics de la matriz H_ij")
end

# ============================================================
# Validación exacta del AutoMPO en el sector de 1 partícula
# sin warning de prime levels
# ============================================================
using ITensors, ITensorMPS, SparseArrays, Random, LinearAlgebra

function one_particle_mps(sites, j::Int)
    N = length(sites)
    st = fill("Emp", N)
    st[j] = "Occ"
    return productMPS(sites, st)
end

# elemento de matriz <i|H|j>, con el bra primado
function mpo_1p_element(mpo, ψbra, ψket)
    return inner(prime(ψbra), mpo, ψket)
end

function validate_autompo_oneparticle(mpo, sites, Hs::SparseMatrixCSC;
        zero_samples::Int=300, atol::Float64=1e-12, seed::Int=1234)

    N = size(Hs, 1)
    I, J, V = findnz(Hs)

    # Base de 1 partícula cacheada
    basis = [one_particle_mps(sites, j) for j in 1:N]

    max_err_nz = 0.0
    n_bad = 0

    println("Validando todos los no nulos de Hs...")
    for t in eachindex(V)
        i, j, v = I[t], J[t], V[t]
        a = mpo_1p_element(mpo, basis[i], basis[j])
        err = abs(a - v)
        max_err_nz = max(max_err_nz, err)
        n_bad += (err > atol)
    end

    rng = MersenneTwister(seed)
    nzset = Set(zip(I, J))

    max_on_zeros = 0.0
    nzero_bad = 0
    ntested = 0
    while ntested < zero_samples
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        (i, j) in nzset && continue
        a = mpo_1p_element(mpo, basis[i], basis[j])
        err = abs(a)
        max_on_zeros = max(max_on_zeros, err)
        nzero_bad += (err > atol)
        ntested += 1
    end

    max_herm = 0.0
    for _ in 1:min(200, zero_samples)
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        aij = mpo_1p_element(mpo, basis[i], basis[j])
        aji = mpo_1p_element(mpo, basis[j], basis[i])
        max_herm = max(max_herm, abs(aij - conj(aji)))
    end

    println("=== Validación AutoMPO en sector 1-partícula ===")
    println("N                  = ", N)
    println("nnz(Hs)            = ", nnz(Hs))
    println("max_err_nonzero    = ", max_err_nz)
    println("n_bad_nonzero      = ", n_bad)
    println("max_abs_on_zeros   = ", max_on_zeros)
    println("n_bad_zeros        = ", nzero_bad)
    println("max_hermiticity    = ", max_herm)

    return (
        N = N,
        nnz = nnz(Hs),
        max_err_nonzero = max_err_nz,
        n_bad_nonzero = n_bad,
        max_abs_on_zeros = max_on_zeros,
        n_bad_zeros = nzero_bad,
        max_hermiticity = max_herm,
    )
end

## COMPARACIONES

using ITensors, ITensorMPS, SparseArrays, Random, LinearAlgebra

# ---------- validación AutoMPO en sector de 1 partícula ----------

function one_particle_mps(sites, j::Int)
    st = fill("Emp", length(sites))
    st[j] = "Occ"
    return productMPS(sites, st)
end

# <i|H|j> con bra primado para evitar warnings de ITensors
mpo_1p_element(mpo, ψi, ψj) = inner(prime(ψi), mpo, ψj)

function validate_autompo_oneparticle(mpo, sites, Hs::SparseMatrixCSC;
        zero_samples::Int=200, atol::Float64=1e-11, seed::Int=1234)

    N = size(Hs, 1)
    I, J, V = findnz(Hs)
    basis = [one_particle_mps(sites, j) for j in 1:N]

    max_err_nz = 0.0
    n_bad_nz = 0
    for t in eachindex(V)
        i, j, v = I[t], J[t], V[t]
        a = mpo_1p_element(mpo, basis[i], basis[j])
        err = abs(a - v)
        max_err_nz = max(max_err_nz, err)
        n_bad_nz += (err > atol)
    end

    rng = MersenneTwister(seed)
    nzset = Set(zip(I, J))
    max_on_zeros = 0.0
    n_bad_zeros = 0
    ntested = 0
    while ntested < zero_samples
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        (i, j) in nzset && continue
        a = mpo_1p_element(mpo, basis[i], basis[j])
        err = abs(a)
        max_on_zeros = max(max_on_zeros, err)
        n_bad_zeros += (err > atol)
        ntested += 1
    end

    max_herm = 0.0
    for _ in 1:min(200, zero_samples)
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        aij = mpo_1p_element(mpo, basis[i], basis[j])
        aji = mpo_1p_element(mpo, basis[j], basis[i])
        max_herm = max(max_herm, abs(aij - conj(aji)))
    end

    return (
        max_abs_err_nonzero = max_err_nz,
        max_abs_on_zeros    = max_on_zeros,
        n_bad_nonzero       = n_bad_nz,
        n_bad_zeros         = n_bad_zeros,
        max_hermiticity_violation = max_herm,
    )
end

# ---------- resumen estructural de AutoMPO ----------

function autompo_structure(out_auto)
    mpo   = out_auto.mpo
    sites = out_auto.sites
    lds   = linkdims(mpo)
    χmax  = isempty(lds) ? 1 : maximum(lds)
    return (
        Nchain   = length(mpo),
        localdim = unique(dim.(sites)),
        linkdims = lds,
        χmax     = χmax,
        nlegs    = [length(inds(mpo[n])) for n in 1:length(mpo)],
    )
end

# ------------------------------------------------------------
# Sparse matrix -> Fermion sites
# ------------------------------------------------------------
"""
    build_fermion_sites(Hs)

Devuelve los siteinds fermiónicos compatibles con Hs.
"""
function build_fermion_sites(Hs::SparseMatrixCSC)
    _check_itensors_autompo()
    N = size(Hs, 1)
    size(Hs, 2) == N || error("Hs debe ser cuadrada")
    return siteinds("Fermion", N)
end

# ------------------------------------------------------------
# Sparse matrix -> MPO many-body
# ------------------------------------------------------------
"""
    build_fermion_mpo_from_sparse(Hs; splitblocks=true, atol=1e-12)

Construye directamente el MPO many-body fermiónico desde Hs.
"""
function build_fermion_mpo_from_sparse(
    Hs::SparseMatrixCSC;
    splitblocks::Bool=true,
    atol::Real=1e-12,
)
    _check_itensors_autompo()

    sites = build_fermion_sites(Hs)
    os    = build_graphene_opsum(Hs; atol=atol)
    mpo   = MPO(os, sites; splitblocks=splitblocks)

    return (Hs=Hs, N=size(Hs, 1), sites=sites, os=os, mpo=mpo)
end

# ------------------------------------------------------------
# Wrapper de usuario: flake
# ------------------------------------------------------------
"""
    build_graphene_flake_fermion_mpo(; radius=5.0, star_amp=0.2,
                                       splitblocks=true, atol=1e-12,
                                       verbose=true)

Construye el MPO many-body del flake de grafeno.
"""
function build_graphene_flake_fermion_mpo(;
    radius::Real=5.0,
    star_amp::Real=0.2,
    splitblocks::Bool=true,
    atol::Real=1e-12,
    verbose::Bool=true,
)
    _check_itensors_autompo()

    h, Hs, N = build_graphene_hamiltonian_sparse(; radius=radius, star_amp=star_amp)

    verbose && println("Construyendo AutoMPO flake: N=$N | nnz(Hs)=$(nnz(Hs))")

    out = build_fermion_mpo_from_sparse(Hs; splitblocks=splitblocks, atol=atol)

    return merge((h=h, radius=radius, star_amp=star_amp), out)
end

# ------------------------------------------------------------
# Wrapper de usuario: torus
# ------------------------------------------------------------
"""
    build_graphene_torus_fermion_mpo(Lx, Ly; t=1.0,
                                     splitblocks=true, atol=1e-12,
                                     verbose=true)

Construye el MPO many-body del honeycomb toroidal periódico.
"""
function build_graphene_torus_fermion_mpo(
    Lx::Int,
    Ly::Int;
    t::Number=1.0,
    splitblocks::Bool=true,
    atol::Real=1e-12,
    verbose::Bool=true,
)
    _check_itensors_autompo()

    Hs = build_honeycomb_torus_hamiltonian_sparse(Lx, Ly; t=t)
    N  = size(Hs, 1)

    verbose && println("Construyendo AutoMPO torus: N=$N | nnz(Hs)=$(nnz(Hs))")

    out = build_fermion_mpo_from_sparse(Hs; splitblocks=splitblocks, atol=atol)

    return merge((Lx=Lx, Ly=Ly, t=t), out)
end

# ------------------------------------------------------------
# Resumen corto
# ------------------------------------------------------------
function pretty_autompo_summary(out)
    println("RESUMEN AUTOMPO")
    haskey(out, :N)      && println("  N        = $(out.N)")
    haskey(out, :radius) && println("  radius   = $(out.radius)")
    haskey(out, :star_amp) && println("  star_amp = $(out.star_amp)")
    haskey(out, :Lx)     && println("  Lx       = $(out.Lx)")
    haskey(out, :Ly)     && println("  Ly       = $(out.Ly)")
    haskey(out, :t)      && println("  t        = $(out.t)")
    haskey(out, :Hs)     && println("  nnz(Hs)  = $(nnz(out.Hs))")
    haskey(out, :sites)  && println("  nsites   = $(length(out.sites))")
    haskey(out, :mpo)    && println("  length(mpo) = $(length(out.mpo))")
    return nothing
end

build_graphene_autompo(Hs::SparseMatrixCSC; kwargs...) =
    build_graphene_opsum(Hs; kwargs...)

build_graphene_flake_autompo(; kwargs...) =
    build_graphene_flake_fermion_mpo(; kwargs...)

# =============================================================================
# SECCIÓN 15 — PATCHING pQTCI (fiel a arXiv:2602.22372)
#
# EL MÉTODO DEL PAPER (sección 2 + Apéndice A):
#
# 1. Se tiene un TT F ≈ F̃ con bond-cap χ_p.
# 2. Si el error max ε > τ (tolerance) Y χ_p está saturado, se SUBDIVIDE:
#    - Se fija el primer índice σ_1 ∈ {1,...,d_1} y se construye un nuevo TT
#      del sufijo F_{σ_1,*,...,*}.
#    - Recursión hasta que ε ≤ τ o la longitud es 1.
# 3. Los patches son TTs más cortos (menores R-ℓ) con χ_p más pequeño.
# 4. La unión de todos los patches cubre el dominio completo.
#
# IMPLEMENTACIÓN AQUÍ:
#   - El split se hace en el PRIMER índice del oráculo completo (nivel ℓ=0).
#   - Para cada prefijo fijo p=(μ_1,...,μ_ℓ), se construye la TCI del sufijo.
#   - La condición de refine: err_est > tol Y χ_max ≥ χtrigger*χ_p.
#   - La evaluación global mezcla los patches por búsqueda de prefijo.
#
# LIMITACIÓN vs PAPER:
#   - El paper describe también la FUSIÓN de patches en un único TT global
#     mediante suma de TTs (cf. Ec. 7-8). Aquí la fusión es opcional y se
#     hace re-ejecutando TCI sobre el oráculo parcheado.
# =============================================================================

struct PatchLeaf
    prefix::Vector{Int}
    carrier::Any          # tci o tt del SUFIJO; nothing si patch trivial
    linkdims::Vector{Int}
    err_est::Float64
    scalar_value::Union{Nothing,Float64}
end

@inline function _prefix_matches(mu, prefix)
    @inbounds for k in eachindex(prefix)
        mu[k] == prefix[k] || return false
    end
    return true
end

function _build_prefix_seed_table(Hperm, R)
    I, J, _ = findnz(Hperm)
    seeds = Dict{Any, Vector{Int}}()
    for t in eachindex(I)
        p = encode_pair(I[t], J[t], R)
        for ℓ in 0:R
            key = ℓ == 0 ? () : Tuple(p[1:ℓ])
            !haskey(seeds, key) && (seeds[key] = copy(p))
        end
    end
    return seeds
end

_has_nonzero(seeds, prefix) = haskey(seeds, Tuple(prefix))

function _restrict_pivots(global_pivots, prefix, seeds, R)
    rem = R - length(prefix)
    rem == 0 && return Vector{Vector{Int}}()
    seen = Set{Any}(); out = Vector{Vector{Int}}()
    for p in global_pivots
        _prefix_matches(p, prefix) || continue
        tail = copy(p[length(prefix)+1:end])
        tp = Tuple(tail)
        tp in seen || (push!(seen, tp); push!(out, tail))
    end
    if isempty(out) && _has_nonzero(seeds, prefix)
        seed = seeds[Tuple(prefix)]
        push!(out, copy(seed[length(prefix)+1:end]))
    end
    return out
end

function _build_patch(Ffull, Hperm, R, global_pivots, seeds, prefix;
        tolerance, patch_maxbonddim, maxiter, do_recompress, recompress_method)
    !_has_nonzero(seeds, prefix) && return PatchLeaf(copy(prefix), nothing, Int[], 0.0, 0.0)
    rem = R - length(prefix)
    rem == 0 && return PatchLeaf(copy(prefix), nothing, Int[], 0.0, Ffull(prefix))

    Fpatch = mu -> Ffull(vcat(prefix, collect(mu)))
    localdims = fill(4, rem)
    pivots = _restrict_pivots(global_pivots, prefix, seeds, R)
    isempty(pivots) && error("Patch no vacío sin pivots: prefix=$prefix")

    T = eltype(Hperm)
    tci, _, err_hist = run_crossinterpolate2(T, Fpatch, localdims, pivots;
        tolerance, maxbonddim=patch_maxbonddim, maxiter)
    canonicalize_if_possible!(tci, Fpatch; tolerance)

    carrier = tci
    tt = to_tensortrain(tci)
    if tt !== nothing
        do_recompress && recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim=patch_maxbonddim)
        carrier = tt
    end

    err_est = isempty(err_hist) ? Inf : float(last(err_hist))
    lds     = collect(safe_linkdims(carrier))
    return PatchLeaf(copy(prefix), carrier, lds, err_est, nothing)
end

_should_refine(leaf, tolerance, χ_p, χtrigger) =
    leaf.scalar_value === nothing &&
    !isempty(leaf.linkdims) &&
    leaf.err_est > tolerance &&
    maximum(leaf.linkdims) >= max(2, floor(Int, χtrigger * χ_p))

function _build_patches_recursive(Ffull, Hperm, R, pivots, seeds, prefix=Int[];
        tolerance, patch_maxbonddim, maxiter, do_recompress, recompress_method, χtrigger=0.95)
    leaf = _build_patch(Ffull, Hperm, R, pivots, seeds, prefix;
        tolerance, patch_maxbonddim, maxiter, do_recompress, recompress_method)

    rem = R - length(prefix)
    if rem == 0 || !_should_refine(leaf, tolerance, patch_maxbonddim, χtrigger)
        return PatchLeaf[leaf]
    end

    out = PatchLeaf[]
    for v in 1:4   # d=4 para ij
        append!(out, _build_patches_recursive(
            Ffull, Hperm, R, pivots, seeds, [prefix; v];
            tolerance, patch_maxbonddim, maxiter, do_recompress, recompress_method, χtrigger))
    end
    return out
end

function _make_patched_eval(leaves, R)
    bylevel = Dict{Int, Dict{Any, PatchLeaf}}()
    for leaf in leaves
        lp   = length(leaf.prefix)
        dict = get!(bylevel, lp, Dict{Any, PatchLeaf}())
        dict[Tuple(leaf.prefix)] = leaf
    end
    levels = sort(collect(keys(bylevel)); rev=true)

    return function(i, j)
        mu = encode_pair(i, j, R)
        for lp in levels
            key  = lp == 0 ? () : Tuple(mu[1:lp])
            dict = bylevel[lp]
            haskey(dict, key) || continue
            leaf = dict[key]
            leaf.scalar_value !== nothing && return leaf.scalar_value
            return safe_evaluate(leaf.carrier, mu[lp+1:end])
        end
        return 0.0
    end
end

"""
    build_graphene_H_mpo_patched(; radius, star_amp, ...)

pQTCI: patching adaptativo del oráculo quantics del flake.
Sigue arXiv:2602.22372 §2 + App. A.
"""
function build_graphene_H_mpo_patched(;
    radius::Real       = 20.0,
    star_amp::Real     = 0.2,
    perm               = nothing,
    tolerance          = DEFAULT_TOL,
    patch_maxbonddim::Int = 64,
    maxiter::Int       = DEFAULT_MAXITER,
    seed::Int          = DEFAULT_SEED,
    pivot_mode::Symbol = :rowcover,
    nsample_pivots::Int = 400,
    extra_pivots::Int   = 300,
    do_recompress::Bool = true,
    recompress_method::Symbol = :CI,
    validate_samples::Bool = true,
    χtrigger::Float64  = 0.95,
)
    h, Hs, N = build_graphene_hamiltonian_sparse(; radius, star_amp)
    perm === nothing && (perm = identity_permutation(N))
    Hperm = apply_site_permutation(Hs, perm)

    Ffull, R, M, Nvalid, T = make_H_oracle(Hperm)
    pivots = choose_initial_pivots_ij(Hperm, R;
        pivot_mode, nsample=nsample_pivots, extra=extra_pivots, seed)
    seeds = _build_prefix_seed_table(Hperm, R)

    leaves = _build_patches_recursive(Ffull, Hperm, R, pivots, seeds;
        tolerance, patch_maxbonddim, maxiter, do_recompress, recompress_method, χtrigger)

    patched_eval = _make_patched_eval(leaves, R)

    patch_chimax = [isempty(l.linkdims) ? 1 : maximum(l.linkdims) for l in leaves]
    patch_levels = [length(l.prefix) for l in leaves]

    patch_val   = validate_samples ? sampled_validation(Hperm, patched_eval; seed=seed+200) : nothing
    all_nz_val  = validate_all_nonzeros(Hperm, patched_eval; seed=seed+300)

    return (h=h, Hs=Hs, Hperm=Hperm, perm=perm, F=Ffull, R=R, M=M, N=N,
            pivots=pivots, leaves=leaves, npatches=length(leaves),
            patch_chimax=patch_chimax, patch_levels=patch_levels,
            patched_eval=patched_eval,
            patch_val=patch_val, all_nz_val=all_nz_val)
end


# nuevas cosas de patching

using LinearAlgebra
using SparseArrays
using Statistics
using Plots

# ============================================================
# Helpers geométricos para patches ij
# ============================================================

"""
Convierte un prefijo quantics ij (mu_1,...,mu_ell), mu_k in 1:4,
en el rectángulo diádico correspondiente dentro de la matriz padded M×M.

Devuelve:
    (row_lo, row_hi, col_lo, col_hi)
con índices 1-based e inclusivos.
"""
function prefix_to_rect_ij(prefix::AbstractVector{<:Integer}, R::Int)
    ell = length(prefix)

    r0 = 0
    c0 = 0
    for μ in prefix
        br, bc = localindex_to_pair(Int(μ))
        r0 = (r0 << 1) | br
        c0 = (c0 << 1) | bc
    end

    blk = 1 << (R - ell)          # tamaño del bloque por eje
    row_lo = r0 * blk + 1
    row_hi = (r0 + 1) * blk
    col_lo = c0 * blk + 1
    col_hi = (c0 + 1) * blk

    return row_lo, row_hi, col_lo, col_hi
end

"""
Reconstruye el evaluador patched global a partir de leaves y R.
Esto replica la lógica de _make_patched_eval del script,
así no dependes de que el named tuple devuelto lo exponga.
"""
function patched_eval_from_leaves(leaves, R::Int)
    bylevel = Dict{Int, Dict{Any, Any}}()
    for leaf in leaves
        lp = length(leaf.prefix)
        dict = get!(bylevel, lp, Dict{Any, Any}())
        dict[Tuple(leaf.prefix)] = leaf
    end
    levels = sort!(collect(keys(bylevel)); rev=true)

    return function(i::Int, j::Int)
        mu = encode_pair(i, j, R)
        for lp in levels
            key  = lp == 0 ? () : Tuple(mu[1:lp])
            dict = bylevel[lp]
            haskey(dict, key) || continue
            leaf = dict[key]
            if leaf.scalar_value !== nothing
                return leaf.scalar_value
            else
                return safe_evaluate(leaf.carrier, mu[lp+1:end])
            end
        end
        return 0.0
    end
end

# ============================================================
# Helpers de matrices dense padded
# ============================================================

"""
Devuelve una matriz dense padded M×M, con M = 2^R.
La Hperm original queda en la esquina superior izquierda.
"""
function dense_padded_H(Hperm::SparseMatrixCSC, R::Int)
    M = 1 << R
    A = zeros(eltype(Hperm), M, M)
    n = size(Hperm, 1)
    A[1:n, 1:n] .= Matrix(Hperm)
    return A
end

"""
Evalúa una función evalfun(i,j) sobre toda la matriz padded M×M.
"""
function dense_from_eval(evalfun, R::Int)
    M = 1 << R
    A = Matrix{Float64}(undef, M, M)
    @inbounds for i in 1:M, j in 1:M
        A[i, j] = float(evalfun(i, j))
    end
    return A
end

"""
Transforma una matriz a escala log10 segura para plot de error.
"""
function log10_safe(A; floor=1e-16)
    return log10.(max.(abs.(A), floor))
end

# ============================================================
# Panel superior: H con rectángulos de patch
# ============================================================

function plot_patches_over_H_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        use_abs::Bool=true,
        title::AbstractString="H padded + patches",
        linewidth_patch::Real=1.0)

    A = dense_padded_H(Hperm, R)
    Z = use_abs ? abs.(A) : real.(A)
    M = size(Z, 1)

    p = heatmap(1:M, 1:M, Z;
        xlabel="columna j",
        ylabel="fila i",
        yflip=true,
        aspect_ratio=:equal,
        title=title,
        colorbar_title = use_abs ? "|H|" : "Re(H)")

    for leaf in leaves
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        xs = [c1, c2, c2, c1, c1]
        ys = [r1, r1, r2, r2, r1]
        plot!(p, xs, ys; lw=linewidth_patch, color=:black, label=false)
    end

    return p
end

# ============================================================
# Panel central: error |H_patch - H|
# ============================================================

function plot_patch_error_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        patched_eval=nothing,
        title::AbstractString="|H_patch - H|",
        logscale::Bool=true,
        floor::Real=1e-16)

    patched_eval === nothing && (patched_eval = patched_eval_from_leaves(leaves, R))

    H_exact = dense_padded_H(Hperm, R)
    H_patch = dense_from_eval(patched_eval, R)
    Err = abs.(H_patch .- H_exact)

    Z = logscale ? log10_safe(Err; floor=floor) : Err
    M = size(Z, 1)

    p = heatmap(1:M, 1:M, Z;
        xlabel="columna j",
        ylabel="fila i",
        yflip=true,
        aspect_ratio=:equal,
        title=title,
        colorbar_title = logscale ? "log10 error" : "error abs")

    for leaf in leaves
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        xs = [c1, c2, c2, c1, c1]
        ys = [r1, r1, r2, r2, r1]
        plot!(p, xs, ys; lw=1.0, color=:black, label=false)
    end

    return p
end

# ============================================================
# Panel inferior: perfiles χ_ell
# ============================================================

"""
Dibuja:
- una curva azul para TCI global ij
- una curva negra discontinua para DAG exacto
- muchas curvas rojas finas para los patches
- opcionalmente una envolvente roja gruesa (máximo patch por corte global)
"""
function plot_bond_profiles_patched_ij(R::Int, leaves;
        out_tci=nothing,
        out_exact=nothing,
        show_patch_envelope::Bool=true,
        title::AbstractString="Perfiles χℓ")

    p = plot(; xlabel="corte global ℓ", ylabel="bond dim χℓ", title=title)

    # curvas de patches
    env = zeros(Int, max(R - 1, 1))
    first_patch = true
    for leaf in leaves
        lp = length(leaf.prefix)
        lds = leaf.linkdims
        isempty(lds) && continue

        xs = collect(lp+1 : lp+length(lds))
        ys = collect(lds)

        plot!(p, xs, ys;
            color=:red, lw=1.5, alpha=0.55,
            label = first_patch ? "patches" : false)
        first_patch = false

        for (x, y) in zip(xs, ys)
            1 <= x <= length(env) || continue
            env[x] = max(env[x], y)
        end
    end

    if show_patch_envelope && any(env .> 0)
        xs = findall(>(0), env)
        ys = env[xs]
        plot!(p, xs, ys;
            color=:darkred, lw=3, label="envolvente patches")
    end

    # TCI global ij
    if out_tci !== nothing
        lds_tci = linkdims_of(out_tci)
        plot!(p, 1:length(lds_tci), lds_tci;
            color=:blue, lw=3, marker=:circle, label="TCI global ij")
    end

    # DAG exacto ij
    if out_exact !== nothing
        lds_exact = out_exact.linkdims
        plot!(p, 1:length(lds_exact), lds_exact;
            color=:black, lw=2, ls=:dash, marker=:diamond, label="DAG exacto ij")
    end

    return p
end

# ============================================================
# Figura completa tipo paper
# ============================================================

function plot_paperstyle_patching_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        patched_eval=nothing,
        out_tci=nothing,
        out_exact=nothing,
        use_abs::Bool=true,
        logscale_error::Bool=true,
        floor_error::Real=1e-16,
        main_title::AbstractString="Patching ij")

    patched_eval === nothing && (patched_eval = patched_eval_from_leaves(leaves, R))

    p1 = plot_patches_over_H_ij(Hperm, R, leaves;
        use_abs=use_abs,
        title = use_abs ? "|H| padded + patches" : "Re(H) padded + patches")

    p2 = plot_patch_error_ij(Hperm, R, leaves;
        patched_eval=patched_eval,
        title="|H_patch - H|",
        logscale=logscale_error,
        floor=floor_error)

    p3 = plot_bond_profiles_patched_ij(R, leaves;
        out_tci=out_tci,
        out_exact=out_exact,
        show_patch_envelope=true,
        title="Perfiles χℓ: global vs patched vs exacto")

    return plot(p1, p2, p3; layout=@layout([a; b; c]), size=(900, 1300), plot_title=main_title)
end


using SparseArrays
using Statistics
using Plots

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

"""
Recorta un rectángulo (r1:r2, c1:c2) al dominio físico 1:N × 1:N.
Devuelve nothing si no intersecta.
"""
function clip_rect_to_N(r1::Int, r2::Int, c1::Int, c2::Int, N::Int)
    rr1 = max(r1, 1)
    rr2 = min(r2, N)
    cc1 = max(c1, 1)
    cc2 = min(c2, N)
    (rr1 <= rr2 && cc1 <= cc2) || return nothing
    return rr1, rr2, cc1, cc2
end

"""
Matriz binaria del soporte de H en tamaño físico N×N.
"""
function support_matrix(Hperm::SparseMatrixCSC)
    N = size(Hperm, 1)
    S = zeros(Float32, N, N)
    I, J, _ = findnz(Hperm)
    @inbounds for t in eachindex(I)
        S[I[t], J[t]] = 1.0f0
    end
    return S
end

"""
Reconstruye H_patch sobre el dominio físico N×N.
"""
function dense_patch_physical(patched_eval, N::Int)
    A = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N, j in 1:N
        A[i, j] = float(patched_eval(i, j))
    end
    return A
end

"""
log10 seguro pero preservando NaN.
"""
function log10_safe_nan(A; floor=1e-16)
    Z = similar(A, Float64)
    @inbounds for i in eachindex(A)
        a = A[i]
        if isnan(a)
            Z[i] = NaN
        else
            Z[i] = log10(max(abs(a), floor))
        end
    end
    return Z
end

# ------------------------------------------------------------
# Plot 1: soporte de H + patches
# ------------------------------------------------------------

"""
Plot más legible que |H| padded:
- usa solo la parte física 1:N × 1:N
- enseña soporte binario
- dibuja patches recortados al dominio físico
"""
function plot_patches_support_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        title::AbstractString = "Soporte(H) + patches",
        linewidth_patch::Real = 1.3)

    N = size(Hperm, 1)
    S = support_matrix(Hperm)

    p = heatmap(
        1:N, 1:N, S;
        xlabel = "columna j",
        ylabel = "fila i",
        yflip = true,
        aspect_ratio = :equal,
        title = title,
        colorbar = false,
    )

    for leaf in leaves
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        clipped = clip_rect_to_N(r1, r2, c1, c2, N)
        clipped === nothing && continue
        rr1, rr2, cc1, cc2 = clipped

        xs = [cc1, cc2, cc2, cc1, cc1]
        ys = [rr1, rr1, rr2, rr2, rr1]
        plot!(p, xs, ys; color=:white, lw=linewidth_patch, label=false)
    end

    return p
end

# ------------------------------------------------------------
# Plot 2: error solo en el soporte físico
# ------------------------------------------------------------

"""
Error |H_patch - H| SOLO donde H != 0.
Fuera del soporte pone NaN para no llenar todo de negro.
"""
function plot_patch_error_on_support_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        patched_eval = nothing,
        logscale::Bool = true,
        floor::Real = 1e-16,
        title::AbstractString = "|H_patch - H| sobre soporte(H)")

    N = size(Hperm, 1)
    patched_eval === nothing && (patched_eval = patched_eval_from_leaves(leaves, R))

    H_exact = Matrix(Hperm)
    H_patch = dense_patch_physical(patched_eval, N)

    Err = fill(NaN, N, N)
    I, J, _ = findnz(Hperm)
    @inbounds for t in eachindex(I)
        i, j = I[t], J[t]
        Err[i, j] = abs(H_patch[i, j] - H_exact[i, j])
    end

    Z = logscale ? log10_safe_nan(Err; floor=floor) : Err

    p = heatmap(
        1:N, 1:N, Z;
        xlabel = "columna j",
        ylabel = "fila i",
        yflip = true,
        aspect_ratio = :equal,
        title = title,
        colorbar_title = logscale ? "log10 error" : "error abs",
    )

    return p
end

# ------------------------------------------------------------
# Figura nueva: sustituye las 2 de arriba
# ------------------------------------------------------------

function plot_paperstyle_patching_ij_v2(Hperm::SparseMatrixCSC, R::Int, leaves;
        patched_eval = nothing,
        out_tci = nothing,
        out_exact = nothing,
        main_title::AbstractString = "Graphene flake — ij patching")

    patched_eval === nothing && (patched_eval = patched_eval_from_leaves(leaves, R))

    p1 = plot_patches_support_ij(Hperm, R, leaves;
        title = "Soporte(H) físico + patches")

    p2 = plot_patch_error_on_support_ij(Hperm, R, leaves;
        patched_eval = patched_eval,
        title = "|H_patch - H| sobre soporte(H)")

    p3 = plot_bond_profiles_patched_ij(R, leaves;
        out_tci = out_tci,
        out_exact = out_exact,
        show_patch_envelope = true,
        title = "Perfiles χℓ: global vs patched vs exacto")

    return plot(p1, p2, p3; layout=@layout([a; b; c]), size=(900, 1300), plot_title=main_title)
end

function plot_paperstyle_patching_ij_v3(Hperm::SparseMatrixCSC, R::Int, leaves;
        out_tci = nothing,
        out_exact = nothing,
        main_title::AbstractString = "Graphene flake — ij patching")

    p1 = plot_patch_depth_map_ij(Hperm, R, leaves;
        title="Mapa de profundidad")
    p2 = plot_patch_chi_map_ij(Hperm, R, leaves;
        title="Mapa de χ_max")
    p3 = plot_bond_profiles_patched_ij(R, leaves;
        out_tci = out_tci,
        out_exact = out_exact,
        show_patch_envelope = true,
        title = "Perfiles χℓ: global vs patched vs exacto")

    return plot(p1, p2, p3; layout=@layout([a; b; c]), size=(900, 1300), plot_title=main_title)
end

using SparseArrays
using Statistics
using Plots

# ============================================================
# Helpers
# ============================================================


"""
χ_max de un leaf:
- 0 si es patch escalar/trivial
- máximo de linkdims si tiene TT
"""
function patch_chimax(leaf)
    if leaf.scalar_value !== nothing
        return 0
    end
    isempty(leaf.linkdims) && return 0
    return maximum(leaf.linkdims)
end

"""
Devuelve una matriz N×N con un valor constante por patch.
Si hay solapamientos al recortar, prevalece el patch más profundo.
Eso es lo natural porque los patches finales son hojas del árbol.
"""
function paint_patch_values_ij(leaves, R::Int, N::Int, valuefun)
    A = fill(NaN, N, N)

    # Pintar primero los más someros y luego los más profundos,
    # para que las hojas profundas sobrescriban al recortar.
    order = sortperm(1:length(leaves), by = k -> length(leaves[k].prefix))

    for k in order
        leaf = leaves[k]
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        clipped = clip_rect_to_N(r1, r2, c1, c2, N)
        clipped === nothing && continue
        rr1, rr2, cc1, cc2 = clipped
        val = valuefun(leaf)
        @views A[rr1:rr2, cc1:cc2] .= val
    end

    return A
end

"""
Dibuja contornos de patches sobre un plot existente.
"""
function overlay_patch_rectangles_ij!(p, leaves, R::Int, N::Int;
        color=:white, lw=1.2, alpha=0.9)

    for leaf in leaves
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        clipped = clip_rect_to_N(r1, r2, c1, c2, N)
        clipped === nothing && continue
        rr1, rr2, cc1, cc2 = clipped

        xs = [cc1, cc2, cc2, cc1, cc1]
        ys = [rr1, rr1, rr2, rr2, rr1]
        plot!(p, xs, ys; color=color, lw=lw, alpha=alpha, label=false)
    end

    return p
end

# ============================================================
# Plot 1: mapa de profundidad
# ============================================================

"""
Mapa de profundidad del patch sobre el dominio físico 1:N × 1:N.
Cada rectángulo toma el valor depth = length(prefix).
"""
function plot_patch_depth_map_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        overlay_rects::Bool=true,
        title::AbstractString="Mapa de profundidad de patches")

    N = size(Hperm, 1)
    D = paint_patch_values_ij(leaves, R, N, leaf -> length(leaf.prefix))

    p = heatmap(
        1:N, 1:N, D;
        xlabel = "columna j",
        ylabel = "fila i",
        yflip = true,
        aspect_ratio = :equal,
        title = title,
        colorbar_title = "depth",
    )

    overlay_rects && overlay_patch_rectangles_ij!(p, leaves, R, N; color=:white, lw=1.0)

    return p
end

# ============================================================
# Plot 2: mapa de χ_max
# ============================================================

"""
Mapa de χ_max del patch sobre el dominio físico 1:N × 1:N.
Cada rectángulo toma el valor χ_max de su leaf.
Los patches escalares quedan en χ=0.
"""
function plot_patch_chi_map_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        overlay_rects::Bool=true,
        title::AbstractString="Mapa de χ_max por patch")

    N = size(Hperm, 1)
    C = paint_patch_values_ij(leaves, R, N, patch_chimax)

    p = heatmap(
        1:N, 1:N, C;
        xlabel = "columna j",
        ylabel = "fila i",
        yflip = true,
        aspect_ratio = :equal,
        title = title,
        colorbar_title = "χ_max",
    )

    overlay_rects && overlay_patch_rectangles_ij!(p, leaves, R, N; color=:white, lw=1.0)

    return p
end

# ============================================================
# Figura combinada: depth + chi
# ============================================================

function plot_patch_maps_ij(Hperm::SparseMatrixCSC, R::Int, leaves;
        main_title::AbstractString="Mapas de patching ij")

    p1 = plot_patch_depth_map_ij(Hperm, R, leaves;
        title="Profundidad del patch")

    p2 = plot_patch_chi_map_ij(Hperm, R, leaves;
        title="χ_max del patch")

    return plot(p1, p2; layout=(1,2), size=(1400, 600), plot_title=main_title)
end

# ============================================================
# Opcional: etiquetas dentro de los rectángulos
# ============================================================

"""
Añade una etiqueta breve en el centro de cada patch.
label_mode = :depth, :chi, o :both
"""
function annotate_patches_ij!(p, Hperm::SparseMatrixCSC, R::Int, leaves;
        label_mode::Symbol = :both,
        min_box_pixels::Int = 40,
        textsize::Int = 7,
        color=:white)

    N = size(Hperm, 1)

    for leaf in leaves
        r1, r2, c1, c2 = prefix_to_rect_ij(leaf.prefix, R)
        clipped = clip_rect_to_N(r1, r2, c1, c2, N)
        clipped === nothing && continue
        rr1, rr2, cc1, cc2 = clipped

        h = rr2 - rr1 + 1
        w = cc2 - cc1 + 1
        max(h, w) < min_box_pixels && continue

        xc = (cc1 + cc2) / 2
        yc = (rr1 + rr2) / 2

        depth = length(leaf.prefix)
        chi   = patch_chimax(leaf)

        label = if label_mode == :depth
            "d=$depth"
        elseif label_mode == :chi
            "χ=$chi"
        else
            "d=$depth\nχ=$chi"
        end

        annotate!(p, xc, yc, text(label, color, textsize, :center))
    end

    return p
end

"""
Versión combinada con etiquetas.
"""
function plot_patch_maps_ij_annotated(Hperm::SparseMatrixCSC, R::Int, leaves;
        main_title::AbstractString="Mapas anotados de patching ij",
        label_mode::Symbol = :both)

    p1 = plot_patch_depth_map_ij(Hperm, R, leaves;
        title="Profundidad del patch")
    annotate_patches_ij!(p1, Hperm, R, leaves; label_mode=:depth)

    p2 = plot_patch_chi_map_ij(Hperm, R, leaves;
        title="χ_max del patch")
    annotate_patches_ij!(p2, Hperm, R, leaves; label_mode=:chi)

    return plot(p1, p2; layout=(1,2), size=(1500, 650), plot_title=main_title)
end


# =============================================================================
# SECCIÓN 16 — POST-COMPRESIÓN SVD DEL MPO EXACTO DAG
#
# El DAG produce el bond dim mínimo EXACTO para la codificación ij dada.
# Sin embargo, numéricamente puede haber valores singulares muy pequeños
# (≪ τ) en los tensores W[ℓ] que se pueden truncar sin perder precisión.
# Esto no reduce el bond dim "matemático" pero sí el "numérico".
#
# Método: SVD de los unfoldings izquierdos W[ℓ] y truncar con rtol.
# =============================================================================

"""
    svd_compress_mpo(W; rtol=1e-13)

Post-comprime un MPO ij (formato [bondL, rowbit, colbit, bondR]) por SVD
de izquierda a derecha. Devuelve (W_compressed, new_linkdims).

No modifica W in-place. Útil para reducir bond dims numéricos del DAG exacto.
"""

function svd_compress_mpo(W::Vector; rtol::Float64=1e-13)
    R  = length(W)
    T  = eltype(W[1])
    Wc = Vector{Array{T,4}}(undef, R)
    for ℓ in 1:R
        Wc[ℓ] = copy(W[ℓ])
    end

    # Sweep R→L: QR para right-orthogonalizar
    for ℓ in R:-1:2
        χL, d1, d2, χR = size(Wc[ℓ])
        # desdobla como (χL, d1*d2*χR) y hace QR de la transpuesta
        M  = reshape(Wc[ℓ], χL, d1 * d2 * χR)  # (χL, resto)
        F  = qr(Matrix(M'))                       # QR de (resto, χL)
        Q  = Matrix(F.Q)                          # (resto, k),  k = min(resto, χL)
        Rm = Matrix(F.R)                          # (k, χL)
        k  = size(Q, 2)
        # core ℓ queda right-orthogonal: shape (χL_new=k, d1, d2, χR)
        # Q es (d1*d2*χR, k), Q' es (k, d1*d2*χR)
        Wc[ℓ] = reshape(Matrix(Q'), k, d1, d2, χR)
        # absorbe Rm' = (χL, k) en el bond derecho del core ℓ-1
        χLp, d1p, d2p, χRp = size(Wc[ℓ-1])
        Mprev  = reshape(Wc[ℓ-1], χLp * d1p * d2p, χRp)  # (χLp*d1p*d2p, χRp)
        # χRp debe coincidir con χL del core ℓ, que es el χL original
        Mprev  = Mprev * Matrix(Rm')                        # (χLp*d1p*d2p, k)
        Wc[ℓ-1] = reshape(Mprev, χLp, d1p, d2p, k)
    end

    # Sweep L→R: SVD con truncación
    for ℓ in 1:R-1
        χL, d1, d2, χR = size(Wc[ℓ])
        M  = reshape(Wc[ℓ], χL * d1 * d2, χR)
        U, s, Vt = svd(M; full=false)
        thr  = rtol * s[1]
        keep = max(1, sum(s .> thr))
        U    = U[:, 1:keep]
        SV   = Diagonal(s[1:keep]) * Vt[1:keep, :]
        Wc[ℓ] = reshape(U, χL, d1, d2, keep)
        χLn, d1n, d2n, χRn = size(Wc[ℓ+1])
        Mn = reshape(Wc[ℓ+1], χLn, d1n * d2n * χRn)
        Wc[ℓ+1] = reshape(SV * Mn, keep, d1n, d2n, χRn)
    end

    new_lds = [size(Wc[ℓ], 4) for ℓ in 1:R-1]
    return Wc, new_lds
end


function _svd_sweep_mpo(W::Vector, rtol::Float64)
    R  = length(W)
    T  = eltype(W[1])
    Wc = deepcopy(W)

    # Left-to-right SVD sweep
    for ℓ in 1:R-1
        χL, d1, d2, χR = size(Wc[ℓ])
        M   = reshape(Wc[ℓ], χL * d1 * d2, χR)
        U, s, Vt = svd(M)
        thr  = rtol * maximum(s)
        keep = max(1, sum(s .> thr))
        U    = U[:, 1:keep]
        sv   = s[1:keep]
        Vt   = Vt[:, 1:keep]'   # (keep, χR)
        Wc[ℓ] = reshape(U, χL, d1, d2, keep)
        # Absorb S*Vt into next core
        χL_next, d1n, d2n, χR_next = size(Wc[ℓ+1])
        Mnext = reshape(Wc[ℓ+1], χL_next, d1n * d2n * χR_next)
        Mnext = Diagonal(sv) * Vt * Mnext
        Wc[ℓ+1] = reshape(Mnext, keep, d1n, d2n, χR_next)
    end
    return Wc
end

"""
    exact_mpo_svd_postcompress(out_exact; rtol=1e-13)

Dado el output de build_sparse_H_mpo_exact, post-comprime por SVD y devuelve
un nuevo out con W_svd y linkdims_svd.
"""
function exact_mpo_svd_postcompress(out_exact; rtol::Float64=1e-13, verbose::Bool=true)
    W_svd, lds_svd = _svd_sweep_mpo(out_exact.W, rtol), nothing
    W_svd2, lds_svd2 = svd_compress_mpo(out_exact.W; rtol)
    R   = out_exact.R
    W_eval_svd = (i,j) -> evaluate_mpo_ij(W_svd2, i, j, R)
    val = sampled_validation(out_exact.Hperm, W_eval_svd)
    lds = [size(W_svd2[ℓ],4) for ℓ in 1:R-1]
    verbose && println("DAG exacto post-SVD (rtol=$rtol):")
    verbose && println("  linkdims originales = $(out_exact.linkdims)")
    verbose && println("  linkdims post-SVD   = $lds")
    verbose && println("  max_abs_err_nz post-SVD = $(val.max_abs_err_nonzero)")
    return (W=W_svd2, linkdims=lds, validation=val, rtol=rtol)
end

# =============================================================================
# SECCIÓN 17 — FUSIÓN EXPLÍCITA DE PATCHES (suma de TTs)
#
# El paper arXiv:2602.22372 Ec. 7-8 describe cómo fusionar un conjunto de
# patches {(prefix_k, TT_k)} en un único TT global mediante la suma directa
# de sus cores:
#
#   F̃(μ) = Σ_k  δ(μ[1:ℓ_k] = prefix_k) · TT_k(μ[ℓ_k+1:R])
#
# La representación MPO de esta suma es un block-diagonal MPO donde:
#   - Los primeros ℓ_k sitios "enrutan" al patch correcto (bond dim = npatches)
#   - Los últimos R-ℓ_k sitios ejecutan el TCI del sufijo
#
# IMPLEMENTACIÓN AQUÍ:
#   Para patches de la misma profundidad ℓ, la suma es literal:
#     W_fused[ℓ] = block-diagonal de los W[ℓ] de cada patch
#   Para patches de diferente profundidad, se necesita padding.
#
# LIMITACIÓN: Esta fusión sólo produce un MPO exactamente si todos los patches
# tienen la misma profundidad. La versión general requiere padding con ceros.
# =============================================================================

"""
    fuse_patches_to_mpo(leaves, R, T)

Fusiona los patches en un único MPO global sumando sus TTs de sufijo.
Solo válido para patches de la misma profundidad (o profundidad 0 = sin patchear).

Devuelve (W_fused, linkdims_fused).
El bond dim del MPO fusionado = suma de bond dims de los patches individuales.
"""
function fuse_patches_to_mpo(leaves::Vector{PatchLeaf}, R::Int, T::Type)
    # Separar patches con carrier (tienen TCI) de los triviales (scalar o vacíos)
    active = [l for l in leaves if l.carrier !== nothing && l.scalar_value === nothing]
    trivial_scalar = [l for l in leaves if l.scalar_value !== nothing && l.scalar_value != 0.0]

    isempty(active) && error("No hay patches activos con TCI para fusionar")

    # Verificar profundidades
    depths = unique([length(l.prefix) for l in active])

    if length(depths) == 1
        depth = depths[1]
        return _fuse_same_depth_patches(active, trivial_scalar, R, depth, T)
    else
        # Fusión general: profundidades mixtas — usamos el método de padding
        return _fuse_mixed_depth_patches(active, trivial_scalar, R, T)
    end
end

function _extract_patch_W(leaf::PatchLeaf, R::Int, T::Type)
    carrier = leaf.carrier
    rem     = R - length(leaf.prefix)
    rawcores = safe_sitetensors(carrier)

    # Inferir orientación con un pivot del carrier
    # Para patches pequeños, usamos infer directo sin test_mus
    sols = _all_chain_layouts(rawcores; physdim=4)
    layouts = sols[1]  # tomamos la primera solución consistente
    cores3  = _build_cores3(rawcores, layouts; physdim=4)

    W_patch = Vector{Array{T,4}}(undef, rem)
    for ell in eachindex(cores3)
        Lb, pd, Rb = size(cores3[ell])
        W_patch[ell] = reshape(Array(cores3[ell]), Lb, 2, 2, Rb)
    end
    return W_patch
end

function _prefix_to_routing_cores(prefix::Vector{Int}, R::Int, n_patches::Int,
                                    patch_idx::Int, T::Type)
    # Para cada nivel ℓ del prefijo, construimos un core de routing:
    # W_route[ℓ] tiene shape (χL, 2, 2, χR) donde solo el "canal" del patch
    # correcto tiene la transición activa.
    depth = length(prefix)
    cores = Vector{Array{T,4}}(undef, depth)
    for ℓ in 1:depth
        br, bc = localindex_to_pair(prefix[ℓ])
        χL = ℓ == 1 ? 1 : n_patches
        χR = ℓ == depth ? 1 : n_patches
        core = zeros(T, χL, 2, 2, χR)
        if ℓ == 1
            core[1, br+1, bc+1, patch_idx] = one(T)
        elseif ℓ == depth
            core[patch_idx, br+1, bc+1, 1] = one(T)
        else
            core[patch_idx, br+1, bc+1, patch_idx] = one(T)
        end
        cores[ℓ] = core
    end
    return cores
end

function _fuse_same_depth_patches(active, trivial_scalar, R, depth, T)
    n = length(active)
    rem = R - depth

    # Extraer W de cada patch activo
    patch_Ws = [_extract_patch_W(l, R, T) for l in active]

    # Bond dims de cada patch en cada nivel del sufijo
    # W_fused[ℓ] para ℓ ≤ depth: enrutamiento (bond dim = n)
    # W_fused[ℓ] para ℓ > depth: block-diagonal de los cores de sufijo

    W_fused = Vector{Array{T,4}}(undef, R)

    # Cores de prefijo: block-diagonal routing
    for ℓ in 1:depth
        # Cada patch tiene un único camino de routing
        χL_fused = ℓ == 1 ? 1 : n
        χR_fused = ℓ == depth ? n : n
        core = zeros(T, χL_fused, 2, 2, χR_fused)
        for k in 1:n
            br, bc = localindex_to_pair(active[k].prefix[ℓ])
            if ℓ == 1
                core[1, br+1, bc+1, k] += one(T)
            else
                core[k, br+1, bc+1, k] += one(T)
            end
        end
        W_fused[ℓ] = core
    end

    # Cores de sufijo: block-diagonal de los cores de cada patch
    for s in 1:rem
        ℓ = depth + s
        # Dimensiones de cada patch en este nivel del sufijo
        χLs = [size(patch_Ws[k][s], 1) for k in 1:n]
        χRs = [size(patch_Ws[k][s], 4) for k in 1:n]
        χL_tot = sum(χLs)
        χR_tot = sum(χRs)

        # Primer nivel de sufijo: conectar con el routing
        if s == 1; χL_tot = n; end

        core = zeros(T, χL_tot, 2, 2, χR_tot)
        off_L = 0; off_R = 0
        for k in 1:n
            χLk = s == 1 ? 1 : χLs[k]
            χRk = χRs[k]
            iL = (s == 1 ? k-1 : off_L) .+ (1:χLk)
            iR = off_R .+ (1:χRk)
            for br in 0:1, bc in 0:1
                core[iL, br+1, bc+1, iR] .+= patch_Ws[k][s][1:χLk, br+1, bc+1, 1:χRk]
            end
            off_L += χLk; off_R += χRk
        end
        W_fused[ℓ] = core
    end

    lds_fused = [size(W_fused[ℓ],4) for ℓ in 1:R-1]
    return W_fused, lds_fused
end

function _fuse_mixed_depth_patches(active, trivial_scalar, R, T)
    # Profundidades mixtas: extendemos cada patch al mismo R con padding
    # Simplificación: convertimos todos al caso depth=0 expandiendo con identidad
    # En la práctica, la fusión real requiere más trabajo; aquí usamos la suma
    # de funciones de evaluación y re-hacemos TCI sobre esa suma.
    # (Esto es equivalente funcionalmente pero no produce un MPO fusionado exacto)
    error("Fusión de patches con profundidades mixtas no implementada. " *
          "Usa recompress_global_mpo_from_patches para re-TCI sobre el oráculo parcheado.")
end

"""
    recompress_global_mpo_from_patches(out_patch; tolerance, maxbonddim, ...)

Re-ejecuta TCI sobre el oráculo parcheado para obtener un MPO único global.
NO es la fusión explícita de TTs del paper (Ec. 7-8), sino una nueva TCI
que usa los patches como oráculo de evaluación eficiente.

La fusión explícita está en fuse_patches_to_mpo (solo para patches de igual profundidad).
"""
function recompress_global_mpo_from_patches(out_patch;
        tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM,
        maxiter=DEFAULT_MAXITER, seed=DEFAULT_SEED,
        pivot_mode=:rowcover, do_recompress=true, recompress_method=:CI,
        validate_samples=true, verbose=true)

    Hperm    = out_patch.Hperm
    R        = out_patch.R
    T        = eltype(Hperm)
    localdims = fill(4, R)
    N        = out_patch.N

    # Oráculo que usa los patches
    Fpatched = function(mu::AbstractVector{<:Integer})
        r, c = decode_mu_ij(mu)
        (r > N || c > N) && return zero(T)
        return out_patch.patched_eval(r, c)
    end

    pivots = choose_initial_pivots_ij(Hperm, R;
        pivot_mode, seed=seed)

    verbose && println("▶ Re-TCI sobre oráculo parcheado...")
    tci, ranks_hist, err_hist = run_crossinterpolate2(T, Fpatched, localdims, pivots;
        tolerance, maxbonddim, maxiter)
    canonicalize_if_possible!(tci, Fpatched; tolerance)

    tt = to_tensortrain(tci)
    if tt !== nothing && do_recompress
        recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim)
    end
    carrier = tt === nothing ? tci : tt

    I_nz,J_nz,_ = findnz(Hperm)
    rng_lay = MersenneTwister(seed+149)
    n_lay   = min(length(I_nz), 512)
    sel_lay = randperm(rng_lay, length(I_nz))[1:n_lay]
    test_mus = [encode_pair(I_nz[t], J_nz[t], R) for t in sel_lay]
    W_global, cores3, layout_err = extract_W_ij(carrier, test_mus)

    W_eval = (i,j) -> evaluate_mpo_ij(W_global, i, j, R)
    W_val  = validate_samples ? sampled_validation(Hperm, W_eval; seed=seed+200) : nothing
    fro    = sqrt(max(frobenius_norm2_mpo_ij(W_global), 0.0))
    lds    = [size(W_global[ℓ],4) for ℓ in 1:R-1]

    verbose && println("  linkdims recomprimido = $lds")
    verbose && println("  χmax = $(maximum(lds))")
    if W_val !== nothing
        verbose && println("  max_abs_err_nz = $(W_val.max_abs_err_nonzero)")
    end

    return (W=W_global, tci=tci, tt=tt, carrier=carrier, linkdims=lds,
            layout_err=layout_err, W_val=W_val, fro_norm=fro,
            ranks_hist=ranks_hist, err_hist=err_hist)
end


function recompress_global_mpo_from_patches_v2(out_patch;
        tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM,
        maxiter=DEFAULT_MAXITER, seed=DEFAULT_SEED,
        do_recompress=true, recompress_method=:CI,
        validate_samples=true, verbose=true)

    Hperm     = out_patch.Hperm
    R         = out_patch.R
    T         = eltype(Hperm)
    localdims = fill(4, R)
    N         = out_patch.N

    # Oráculo parcheado (exacto)
    Fpatched = function(mu::AbstractVector{<:Integer})
        r, c = decode_mu_ij(mu)
        (r > N || c > N) && return zero(T)
        return out_patch.patched_eval(r, c)
    end

    # ── Pivots desde los patches, no desde rowcover ──────────────────────
    # Cada patch ya encontró sus pivots óptimos para su subdominio.
    # Los concatenamos como pivots globales: prefijo + pivots del sufijo.
    seen   = Set{Any}()
    pivots = Vector{Vector{Int}}()
    for leaf in out_patch.leaves
        leaf.carrier === nothing && continue
        isempty(leaf.linkdims)   && continue
        depth = length(leaf.prefix)
        # Reconstruir algunos pivots globales desde el carrier del patch
        # usando el primer pivot del sufijo que sabemos que es no nulo
        try
            # safe_sitetensors da los cores; el primer pivot es el índice
            # del primer core no nulo del sufijo
            I_nz, J_nz, _ = findnz(Hperm)
            for t in eachindex(I_nz)
                p = encode_pair(I_nz[t], J_nz[t], R)
                _prefix_matches(p, leaf.prefix) || continue
                tp = Tuple(p)
                tp in seen || (push!(seen, tp); push!(pivots, collect(p)))
                break
            end
        catch; end
    end
    isempty(pivots) && (pivots = choose_initial_pivots_ij(Hperm, R;
        pivot_mode=:rowcover, seed=seed))

    verbose && println("▶ Re-TCI v2: $(length(pivots)) pivots desde patches")

    tci, ranks_hist, err_hist = run_crossinterpolate2(T, Fpatched, localdims, pivots;
        tolerance, maxbonddim, maxiter)
    canonicalize_if_possible!(tci, Fpatched; tolerance)

    tt = to_tensortrain(tci)
    if tt !== nothing && do_recompress
        recompress_tt!(tt; method=recompress_method, tolerance, maxbonddim)
    end
    carrier = tt === nothing ? tci : tt

    I_nz, J_nz, _ = findnz(Hperm)
    rng_lay  = MersenneTwister(seed+149)
    n_lay    = min(length(I_nz), 512)
    sel_lay  = randperm(rng_lay, length(I_nz))[1:n_lay]
    test_mus = [encode_pair(I_nz[t], J_nz[t], R) for t in sel_lay]
    W_global, cores3, layout_err = extract_W_ij(carrier, test_mus)

    W_eval = (i,j) -> evaluate_mpo_ij(W_global, i, j, R)
    W_val  = validate_samples ? sampled_validation(Hperm, W_eval; seed=seed+200) : nothing
    lds    = [size(W_global[ℓ], 4) for ℓ in 1:R-1]

    verbose && println("  linkdims = $lds")
    verbose && println("  χmax     = $(maximum(lds))")
    W_val !== nothing &&
        verbose && println("  max_err  = $(W_val.max_abs_err_nonzero)")

    return (W=W_global, tci=tci, tt=tt, carrier=carrier,
            linkdims=lds, W_val=W_val, layout_err=layout_err)
end

# =============================================================================
# SECCIÓN 18 — MODELO DE REFERENCIA: CADENA 1D PERIÓDICA
#
# Para entender por qué torus tiene menor bond dim que flake,
# añadimos un modelo aún más simple: cadena 1D con PBC.
# H_{ij} = -t (δ_{j,i+1} + δ_{j,i-1}) con PBC.
# Este es el caso más compresible: bond dim exacto = 2 para todo R.
#
# También añadimos cadena 1D CON CONTORNO (open boundary, OBC):
# los sitios extremos solo tienen un vecino.
# =============================================================================

"""
    build_chain_hamiltonian_sparse(N; t=1.0, pbc=true)

Cadena tight-binding 1D de N sitios.
pbc=true: condiciones periódicas (torus 1D), bond dim quantics = 2.
pbc=false: contorno abierto, bond dim quantics ≈ 2 también (NN puro).
"""
function build_chain_hamiltonian_sparse(N::Int; t::Number=1.0, pbc::Bool=true)
    τ = float(t)
    T = promote_type(typeof(τ), typeof(conj(τ)))
    rows, cols, vals = Int[], Int[], T[]
    for i in 1:N
        j = mod1(i+1, N)
        (pbc || i < N) || continue
        push!(rows, i); push!(cols, j); push!(vals, -τ)
        push!(rows, j); push!(cols, i); push!(vals, -conj(τ))
    end
    H = sparse(rows, cols, vals, N, N)
    dropzeros!(H)
    return H
end

"""
    build_chain_H_mpo_exact(N; t=1.0, pbc=true, kwargs...)

MPO exacto DAG + TCI para una cadena 1D.
Útil para verificar que:
  - Con PBC: bond dim exacto = 2 (solo hopping NN con wraparound)
  - Con OBC: bond dim exacto ≈ 2 también
  - TCI converge al mismo bond dim con tau pequeño
"""
function build_chain_H_mpo_exact(N::Int; t::Real=1.0, pbc::Bool=true,
        validate_samples::Bool=true, seed::Int=DEFAULT_SEED, verbose::Bool=true)
    H   = build_chain_hamiltonian_sparse(N; t, pbc)
    out = build_sparse_H_mpo_exact(H; validate_samples, seed, verbose)
    return merge(out, (pbc=pbc, t=t))
end

function build_chain_H_mpo_tci(N::Int; t::Real=1.0, pbc::Bool=true,
        tolerance=DEFAULT_TOL, maxbonddim=DEFAULT_MAXBONDDIM, maxiter=DEFAULT_MAXITER,
        seed=DEFAULT_SEED, pivot_mode=:rowcover,
        do_recompress=true, recompress_method=:CI, validate_samples=true)
    H    = build_chain_hamiltonian_sparse(N; t, pbc)
    perm = identity_permutation(N)
    return _run_tci_ij_pipeline(H, H, perm;
        tolerance, maxbonddim, maxiter, seed, pivot_mode,
        do_recompress, recompress_method, validate_samples)
end

# =============================================================================
# SECCIÓN 19 — PLOTS
# =============================================================================

function plot_sparse_pattern(Hperm; markersize=1.2, title="Patrón sparse de Hperm")
    I, J, _ = findnz(Hperm)
    scatter(J, I; markersize, markerstrokewidth=0, marker=:rect,
        xlabel="columna j", ylabel="fila i", title, legend=false,
        yflip=true, aspect_ratio=:equal)
end

function plot_bond_dimensions(out; label="")
    lds = linkdims_of(out)
    plot(1:length(lds), lds; marker=:o, xlabel="corte ℓ",
        ylabel="bond dim χ_ℓ", title="Bond dimensions del MPO",
        legend=!isempty(label), label=label)
end

function plot_bond_dims_comparison(outs_named::Vector{<:Tuple}; title="Comparación bond dims")
    p = plot(; xlabel="corte ℓ", ylabel="bond dim χ_ℓ", title)
    for (label, out) in outs_named
        lds = linkdims_of(out)
        plot!(p, 1:length(lds), lds; marker=:o, label=label)
    end
    return p
end

function plot_nz_offset_hist(Hperm; bins=100)
    I, J, _ = findnz(Hperm)
    histogram(J .- I; bins, xlabel="j-i", ylabel="frecuencia",
        title="Offsets no nulos", legend=false)
end

# =============================================================================
# SECCIÓN 20 — RESÚMENES Y COMPARACIÓN FINAL
# =============================================================================

function print_comparison_table(results::Vector{<:NamedTuple})
    println("\n" * "="^80)
    println(rpad("Caso",30), rpad("N",8), rpad("R",5), rpad("χmax",8), rpad("W_max_err_nz",18))
    println("-"^80)
    for r in results
        χmax  = hasproperty(r, :χmax)      ? r.χmax      :
                hasproperty(r, :chimax)    ? r.chimax    : "?"
        err   = hasproperty(r, :W_val)     && r.W_val !== nothing ? r.W_val.max_abs_err_nonzero :
                hasproperty(r, :sampled_validation) && r.sampled_validation !== nothing ? r.sampled_validation.max_abs_err_nonzero : "—"
        println(rpad(string(r.label),30), rpad(r.N,8), rpad(r.R,5), rpad(string(χmax),8), rpad(string(err),18))
    end
    println("="^80)
end

function quick_summary(label, out)
    lds  = linkdims_of(out)
    χmax = maximum(lds)
    println("\n--- $label ---")
    println("  N = $(out.N) | R = $(out.R) | χmax = $χmax")
    if hasproperty(out, :W_val) && out.W_val !== nothing
        v = out.W_val
        println("  max_abs_err_nz  = $(v.max_abs_err_nonzero)")
        println("  max_abs_zeros   = $(v.max_abs_on_sampled_zeros)")
        println("  max_hermiticity = $(v.max_hermiticity_violation)")
    end
end

# =============================================================================
# SECCIÓN — RANGO SVD EXACTO DEL UNFOLDING QUANTICS
#
# El rango del unfolding en el corte ℓ es el bond dim mínimo que puede tener
# CUALQUIER representación TT exacta de H en ese corte.
# Es un invariante de H, no del método (TCI, DAG, etc.).
#
# Unfolding ij en corte ℓ:
#   filas    → índices quantics μ_1,...,μ_ℓ  (4^ℓ posibles)
#   columnas → índices quantics μ_{ℓ+1},...,μ_R  (4^{R-ℓ} posibles)
#   entrada  → H[r,c] para el (r,c) que codifica ese (μ_1,...,μ_R)
#
# Para matrices grandes NO construimos la matriz densa del unfolding.
# Usamos que el unfolding es sparse: solo hay nnz(H) entradas no nulas.
# El rango se calcula como rango de la Gram matrix A·Aᵀ o Aᵀ·A
# (la más pequeña), que tiene tamaño min(4^ℓ, 4^{R-ℓ}) × mismo.
#
# LÍMITE PRÁCTICO: para ℓ tal que min(4^ℓ, 4^{R-ℓ}) > ~4000,
# la Gram es demasiado grande. La función lo detecta y devuelve `missing`.
# Para radius=5  (N≈100,  R=7): todos los cortes son manejables.
# Para radius=10 (N≈400,  R=9): cortes centrales tienen Gram ~1000×1000, OK.
# Para radius=20 (N≈3000, R=12): cortes centrales tienen Gram ~4^6=4096×4096,
#   pesado pero posible; ponemos max_gram_size=4000 por defecto.
# =============================================================================
 
"""
    quantics_unfolding_sparse(Hperm, R, ell)
 
Construye la matriz del unfolding quantics ij en el corte `ell` como
SparseMatrixCSC. Filas = prefijos μ_1...μ_ell, columnas = sufijos μ_{ell+1}...μ_R.
"""
function quantics_unfolding_sparse(Hperm::SparseMatrixCSC, R::Int, ell::Int)
    1 <= ell < R || error("ell debe satisfacer 1 ≤ ell < R, got ell=$ell R=$R")
    nrow = 4^ell
    ncol = 4^(R - ell)
    I_nz, J_nz, V_nz = findnz(Hperm)
    N = size(Hperm, 1)
 
    rows = Vector{Int}(undef, length(V_nz))
    cols = Vector{Int}(undef, length(V_nz))
 
    @inbounds for t in eachindex(V_nz)
        mu = encode_pair(I_nz[t], J_nz[t], R)
        # índice de fila = base-4 de μ_1...μ_ell (1-based)
        r = 1
        for k in 1:ell;     r = 4*(r-1) + mu[k]; end
        # índice de columna = base-4 de μ_{ell+1}...μ_R (1-based)
        c = 1
        for k in ell+1:R;   c = 4*(c-1) + mu[k]; end
        rows[t] = r
        cols[t] = c
    end
 
    return sparse(rows, cols, V_nz, nrow, ncol)
end
 
"""
    svd_rank_of_unfolding(A; rtol=1e-12)
 
Calcula el rango numérico de la matriz sparse A usando la Gram matrix
(A·Aᵀ si nrow ≤ ncol, Aᵀ·A si ncol < nrow).
Más eficiente que SVD directo cuando una dimensión es mucho menor que la otra.
"""
function svd_rank_of_unfolding(A::SparseMatrixCSC; rtol::Float64=1e-12)
    m, n = size(A)
    if m <= n
        G = Hermitian(Matrix(A * A'))
    else
        G = Hermitian(Matrix(A' * A))
    end
    λ = eigvals(G)
    λmax = maximum(abs, λ)
    λmax == 0.0 && return 0
    return count(x -> abs(x) > rtol^2 * λmax, λ)
end
 
"""
    exact_svd_bond_ranks(Hperm, R; rtol=1e-12, max_gram_size=4000, verbose=true)
 
Calcula el rango SVD exacto del unfolding quantics ij en cada corte ℓ = 1,...,R-1.
Devuelve un Vector donde la entrada ℓ es el rango en ese corte,
o `missing` si la Gram sería demasiado grande (> max_gram_size).
 
Interpretación: este es el bond dim mínimo que CUALQUIER TT exacto de H puede tener.
"""
function exact_svd_bond_ranks(Hperm::SparseMatrixCSC, R::Int;
        rtol::Float64=1e-12,
        max_gram_size::Int=4000,
        verbose::Bool=true)
 
    ranks = Union{Int,Missing}[]
    for ell in 1:R-1
        nrow = 4^ell
        ncol = 4^(R - ell)
        gram_size = min(nrow, ncol)
        if gram_size > max_gram_size
            verbose && println("  corte ℓ=$ell: Gram $(gram_size)×$(gram_size) > max_gram_size=$max_gram_size → saltado")
            push!(ranks, missing)
            continue
        end
        A = quantics_unfolding_sparse(Hperm, R, ell)
        rk = svd_rank_of_unfolding(A; rtol=rtol)
        verbose && println("  corte ℓ=$ell: nrow=$nrow ncol=$ncol gram=$(gram_size)×$(gram_size) → rango=$rk")
        push!(ranks, rk)
    end
    return ranks
end
 
"""
    compute_and_plot_svd_ranks(Hs; radius, star_amp, rtol, max_gram_size)
 
Pipeline completo: construye H del flake, calcula rangos SVD exactos,
compara con TCI y DAG si se pasan como argumentos opcionales, y hace el plot.
 
Uso mínimo (solo rangos exactos):
    compute_and_plot_svd_ranks(Hs)
 
Uso completo (con comparación):
    compute_and_plot_svd_ranks(Hs; out_tci=out_flake_ij, out_dag=out_flake_exact)
"""
function compute_and_plot_svd_ranks(Hs::SparseMatrixCSC;
        rtol::Float64      = 1e-12,
        max_gram_size::Int = 4000,
        out_tci            = nothing,
        out_dag            = nothing,
        verbose::Bool      = true,
)
    N = size(Hs, 1)
    R = max(1, ceil(Int, log2(max(N, 1))))
    Hperm = Hs   # identidad; si quieres otra permutación pásala antes
 
    verbose && println("N=$N | R=$R | nnz=$(nnz(Hs))")
    verbose && println("Calculando rangos SVD exactos del unfolding quantics ij...")
 
    ranks = exact_svd_bond_ranks(Hperm, R; rtol=rtol,
                                  max_gram_size=max_gram_size, verbose=verbose)
 
    # ── plot ──────────────────────────────────────────────────────────────────
    ells_known = [ell for ell in 1:R-1 if !ismissing(ranks[ell])]
    rks_known  = [ranks[ell] for ell in ells_known]
 
    p = plot(; xlabel="corte ℓ", ylabel="bond dim / rango",
               title="Rango SVD exacto del unfolding (N=$N, R=$R)",
               legend=:topright)
 
    plot!(p, ells_known, rks_known;
          marker=:circle, lw=2, color=:black, label="rango SVD exacto")
 
    if out_tci !== nothing
        lds_tci = linkdims_of(out_tci)
        plot!(p, 1:length(lds_tci), lds_tci;
              marker=:square, lw=2, color=:blue, label="TCI ij")
    end
 
    if out_dag !== nothing
        lds_dag = out_dag.linkdims
        plot!(p, 1:length(lds_dag), lds_dag;
              marker=:diamond, lw=2, color=:red, ls=:dash, label="DAG exacto (sin SVD)")
    end
 
    verbose && println()
    verbose && println("Rangos SVD exactos por corte:")
    for ell in 1:R-1
        r = ranks[ell]
        println("  ℓ=$ell → ", ismissing(r) ? "SALTADO (Gram demasiado grande)" : r)
    end
 
    return (ranks=ranks, R=R, N=N, plot=p)
end


# =============================================================================
# SECCIÓN HIGH-LEVEL — QuanticsTCI.jl (alto nivel sobre TCI.jl)
#
# Requiere: ] add QuanticsTCI
#
# QuanticsTCI.jl hace exactamente lo mismo que nuestro pipeline manual pero
# con una API de una sola línea. Internamente usa TCI.crossinterpolate2 con
# la codificación quantics interleaved (nuestro "ij") o fused.
#
# Diferencias con nuestro bajo nivel:
#   - La codificación quantics la gestiona el paquete (no tú)
#   - Los pivots iniciales se eligen automáticamente
#   - El output es un QuanticsTensorTrain con métodos de evaluación propios
#   - El gauge canónico se aplica por defecto
# =============================================================================

const _HAVE_QTCI = try
    using QuanticsTCI
    true
catch
    false
end

"""
    H_mpo_highlevel_ij(Hs; tolerance, maxbonddim, maxiter, seed)

Versión de alto nivel usando QuanticsTCI.jl.
Equivalente a build_graphene_H_mpo_ij pero con API de una línea.

La codificación 'interleaved' de QuanticsTCI es exactamente nuestra ij:
  μ_ℓ ↔ (bit_ℓ(row), bit_ℓ(col))  con dim local = 4
"""

function build_H_mpo_highlevel_ij(Hs::SparseMatrixCSC;
        tolerance   = DEFAULT_TOL,
        maxbonddim  = DEFAULT_MAXBONDDIM,
        maxiter     = DEFAULT_MAXITER,
        seed        = DEFAULT_SEED,
        verbose     = true)

    _HAVE_QTCI || error("QuanticsTCI no disponible.")

    N = size(Hs, 1)
    T = eltype(Hs)
    R = max(1, ceil(Int, log2(max(N, 1))))
    M = 1 << R

    f_matrix = (i, j) -> begin
        ii = round(Int, i); jj = round(Int, j)
        (ii < 1 || jj < 1 || ii > N || jj > N) && return zero(T)
        Hs[ii, jj]
    end

    xvals = [collect(1:M), collect(1:M)]

    # ── Pivots iniciales desde no-nulos de H ─────────────────────────────
    # QuanticsTCI espera pivots como Vector{Vector{Float64}} donde cada
    # vector interno tiene longitud = número de argumentos (2 aquí),
    # con los VALORES del grid (no los índices quantics).
    I_nz, J_nz, _ = findnz(Hs)
    rng = MersenneTwister(seed)
    n_piv = min(50, length(I_nz))
    sel   = randperm(rng, length(I_nz))[1:n_piv]
    # Los valores del grid son Float64 porque DiscretizedGrid usa floats
    initialpivots = [[Float64(I_nz[t]), Float64(J_nz[t])] for t in sel]

    verbose && println("▶ QuanticsTCI alto nivel: N=$N R=$R M=$M")
    verbose && println("   n_pivots iniciales = $(length(initialpivots))")

    qtt, ranks, errors = QuanticsTCI.quanticscrossinterpolate(
        T, f_matrix, xvals, initialpivots;   # ← pivots como 4º arg posicional
        tolerance       = tolerance,
        maxbonddim      = maxbonddim,
        maxiter         = maxiter,
        unfoldingscheme = :interleaved,
        verbosity       = verbose ? 1 : 0,
    )

    # ── Extraer linkdims del objeto interno ──────────────────────────────
    inner = if hasproperty(qtt, :tci)
        qtt.tci
    elseif hasproperty(qtt, :tensorci2)
        qtt.tensorci2
    else
        error("Campos de QuanticsTensorCI2: $(propertynames(qtt))")
    end

    lds  = collect(TCI.linkdims(inner))
    χmax = maximum(lds)
    verbose && println("   linkdims = $lds")
    verbose && println("   χmax     = $χmax")

    val_hl = sampled_validation(Hs,
        (i, j) -> QuanticsTCI.evaluate(qtt, [i, j]))
    verbose && println("   max_err_nz = $(val_hl.max_abs_err_nonzero)")

    return (qtt=qtt, inner=inner, R=R, M=M, N=N, χmax=χmax, linkdims=lds,
            ranks=ranks, errors=errors, val=val_hl)
end

"""
El cambio clave es que `initialpivots` se pasa como **cuarto argumento posicional**, no como keyword. La firma de `quanticscrossinterpolate` es:
"""
#quanticscrossinterpolate(T, f, xvals, initialpivots; kwargs...)


"""
    build_H_mpo_highlevel_fused(Hs; tolerance, maxbonddim, maxiter)

Variante 'fused' (= nuestro ijkl): agrupa bits de fila juntos, bits de columna juntos.
La dim local es 4 pero el agrupamiento es distinto al interleaved.

Nota: en QuanticsTCI.jl 'fused' significa R/2 sitios con dim local 4 cada uno,
donde cada sitio lleva (bit_row_ℓ, bit_col_ℓ) sin interleaving.
"""
function build_H_mpo_highlevel_fused(Hs::SparseMatrixCSC;
        tolerance   = DEFAULT_TOL,
        maxbonddim  = DEFAULT_MAXBONDDIM,
        maxiter     = DEFAULT_MAXITER,
        verbose     = true)

    _HAVE_QTCI || error("QuanticsTCI no disponible.")
    N = size(Hs, 1)
    T = eltype(Hs)

    f_matrix = (i, j) -> begin
        (i > N || j > N) && return zero(T)
        Hs[i, j]
    end

    qtt, ranks, errors = QuanticsTCI.quanticscrossinterpolate(
        T,
        f_matrix,
        [N, N];
        tolerance       = tolerance,
        maxbonddim      = maxbonddim,
        maxiter         = maxiter,
        unfoldingscheme = :fused,   # agrupa todos los bits de fila / todos los de columna
        verbosity       = verbose ? 1 : 0,
    )

    χmax = maximum(TCI.linkdims(qtt))
    eval_hl = (i, j) -> QuanticsTCI.evaluate(qtt, i, j)
    val_hl  = sampled_validation(Hs, eval_hl)

    return (qtt=qtt, R=length(TCI.linkdims(qtt))+1, N=N, χmax=χmax,
            ranks=ranks, errors=errors, val=val_hl)
end

"""
    compare_lowlevel_vs_highlevel(Hs; kwargs...)

Compara nuestro pipeline bajo nivel con QuanticsTCI.jl alto nivel.
Útil para verificar que ambos obtienen el mismo bond dim y error.
"""
function compare_lowlevel_vs_highlevel(Hs::SparseMatrixCSC;
        tolerance=1e-11, maxbonddim=256, maxiter=15, seed=DEFAULT_SEED)

    _HAVE_QTCI || error("QuanticsTCI no disponible.")
    N = size(Hs, 1)

    println("=== Comparación bajo nivel vs alto nivel ===")
    println("N=$N | nnz=$(nnz(Hs))")
    println()

    t_low = @elapsed begin
        perm = identity_permutation(N)
        out_low = _run_tci_ij_pipeline(Hs, Hs, perm;
            tolerance, maxbonddim, maxiter, seed,
            validate_samples=true, do_recompress=true)
    end

    t_high = @elapsed begin
        out_high = build_H_mpo_highlevel_ij(Hs;
            tolerance, maxbonddim, maxiter, seed, verbose=false)
    end

    lds_low  = linkdims_of(out_low)
    lds_high = TCI.linkdims(out_high.qtt)

    println(rpad("Método", 20), rpad("χmax", 8), rpad("max_err_nz", 18), rpad("tiempo(s)", 10))
    println("-"^60)
    println(rpad("Bajo nivel (manual)", 20),
            rpad(maximum(lds_low), 8),
            rpad(string(round(out_low.W_val.max_abs_err_nonzero, sigdigits=3)), 18),
            rpad(round(t_low, sigdigits=3), 10))
    println(rpad("Alto nivel (QTCI.jl)", 20),
            rpad(maximum(lds_high), 8),
            rpad(string(round(out_high.val.max_abs_err_nonzero, sigdigits=3)), 18),
            rpad(round(t_high, sigdigits=3), 10))
    println()
    println("linkdims bajo nivel : $lds_low")
    println("linkdims alto nivel : $(collect(lds_high))")

    return (low=out_low, high=out_high, t_low=t_low, t_high=t_high)
end

# EVALUADOR
using SparseArrays, LinearAlgebra, Random

"""
Devuelve una función evalA(i,j) que evalúa la entrada (i,j)
para cualquiera de tus salidas:
- patching         -> out.patched_eval
- MPO ij global    -> evaluate_mpo_ij(out.W, ...)
- MPO ijkl global  -> evaluate_mpo_ijkl(out.W, ...)
- carrier TT/TCI   -> safe_evaluate(...)

NOTA:
- para patching, esto usa la suma implícita por prefijos;
- para métodos globales usa el MPO/TT ya extraído.
"""
function evalfun_from_out(out)
    if hasproperty(out, :patched_eval)
        return (i::Int, j::Int) -> out.patched_eval(i, j)
    elseif hasproperty(out, :W)
        W = out.W
        if ndims(W[1]) == 4
            return (i::Int, j::Int) -> evaluate_mpo_ij(W, i, j, out.R)
        elseif ndims(W[1]) == 6
            side_pad = hasproperty(out, :side_pad) ? out.side_pad :
                       error("Falta side_pad para evaluar MPO ijkl")
            return (i::Int, j::Int) -> evaluate_mpo_ijkl(W, i, j, out.R, side_pad)
        else
            error("No reconozco el layout de W")
        end
    elseif hasproperty(out, :carrier) && hasproperty(out, :R)
        return (i::Int, j::Int) -> safe_evaluate(out.carrier, encode_pair(i, j, out.R))
    else
        error("No sé cómo evaluar este objeto")
    end
end
# RECUPERAR MATRIZ DE NUEVO DESDE TT/MPO
"""
Reconstruye la matriz completa N×N evaluando el MPO/TT entrada a entrada.
Úsalo solo para tamaños moderados.
"""
function recover_dense_matrix_from_out(out; N::Int=out.N, T=ComplexF64)
    evalA = evalfun_from_out(out)
    A = Matrix{T}(undef, N, N)
    @inbounds for i in 1:N, j in 1:N
        A[i, j] = evalA(i, j)
    end
    return A
end

"""
Reconstruye solo en el soporte conocido de una matriz sparse Aref.
Esto es mucho más barato y suele ser la forma correcta de validar.
"""
function recover_on_known_support(out, Aref::SparseMatrixCSC)
    evalA = evalfun_from_out(out)
    I, J, _ = findnz(Aref)
    vals = Vector{ComplexF64}(undef, length(I))
    @inbounds for t in eachindex(I)
        vals[t] = evalA(I[t], J[t])
    end
    return sparse(I, J, vals, size(Aref,1), size(Aref,2))
end

"""
Reconstrucción sparse por barrido completo + umbral.
Úsalo solo si N no es grande.
"""
function recover_sparse_threshold_from_out(out; N::Int=out.N, atol::Real=1e-12)
    evalA = evalfun_from_out(out)
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    @inbounds for i in 1:N, j in 1:N
        v = evalA(i, j)
        if abs(v) > atol
            push!(rows, i); push!(cols, j); push!(vals, v)
        end
    end
    return sparse(rows, cols, vals, N, N)
end

# Evaluador generico desde cualquiera de las salidas













# =============================================================================
# FIN DE graphene_qtci_all.jl
# =============================================================================
