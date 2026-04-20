# =============================================================================
# mpo_small_systems.jl
# =============================================================================
# MPO vía QTCI para múltiples geometrías de red tight-binding.
#
# Referencia: Núñez Fernández et al., arXiv:2407.02454
#   Sec. 7:     Construcción de MPOs
#   Sec. 4.3.5: Pivots globales
#   Sec. 4.3.6: Ergodicidad — matrices sparse necesitan pivots globales
#
# NOTA Quantica:
#   supercell(region=...) → sistema finito (0D), sin periodicidad → h(()) funciona
#   supercell(N)          → mantiene periodicidad → h(()) FALLA, necesita h((k,)) con k vector de Bloch → no sirve para MPOs

#   Sistemas con supercell(region=...): (OBC-bordes abiertos)
#   Sistemas a mano (matriz explicita): (PBC-bordes periódicos sin Bloch phases): torus y cadenas con potencial on-site se construyen sin Quantica para evitar problemas de Bloch phases y tener control total sobre la matriz. Para estos.

using LinearAlgebra, SparseArrays, Printf, Random
using Quantica
import TensorCrossInterpolation as TCI

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CODIFICACIÓN QUANTICS ij  (d = 4)
# ═══════════════════════════════════════════════════════════════════════════════
# Sec. 7.1: H[i,j] se codifica como tensor F(μ₁,...,μ_R) con
#   μ_ℓ = 1 + bit_ℓ(i) + 2·bit_ℓ(j)  ∈ {1,2,3,4}
#   R = ⌈log₂ N⌉

@inline _p2l(bi, bj) = 1 + bi + 2bj # pair to local (de los 2 bits originales, construye el número quantics: 1,2,3,4)
@inline _l2p(mu) = (y = mu - 1; (y & 1, (y >> 1) & 1)) # local to pair (de 1,2,3,4 recupera los 2 bits originales)
_bits(idx, R) = [Int((idx - 1) >> (R - ℓ)) & 1 for ℓ in 1:R] # extrae los R bits del entero idx, en formato big-endian (bit más significativo primero)

const DEFAULT_TOL = 1e-10
const DEFAULT_MAXITER = 2000
#const DEFAULT_MAXBONDDIM = 64

# dado elemento de una matriz (i,j), devuelve el vector quantics μ con R componentes
function encode_pair(i::Int, j::Int, R::Int)
    ib = _bits(i, R); jb = _bits(j, R)
    [_p2l(ib[ℓ], jb[ℓ]) for ℓ in 1:R]
end

# dado un vector quantics μ, devuelve el par (i,j) decodificado
function decode_pair(mu::AbstractVector{<:Integer})
    r = 0; c = 0
    for x in mu; bi, bj = _l2p(x); r = (r << 1) | bi; c = (c << 1) | bj; end
    (r + 1, c + 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ORÁCULO, PIVOTS, RECONSTRUCCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Dada una matriz A, devuelve un oráculo F(μ) que evalúa A[i,j] a partir de μ codificado.
# Prepara la funcion que TCI va a interpolar (el oráculo: TCI solo necesita poder evaluar F en puntos concretos, sin ver toda la matriz)
function make_oracle(A::AbstractMatrix)
    N = size(A, 1)
    R = max(1, ceil(Int, log2(max(N, 1))))
    T = eltype(A) <: Real ? Float64 : ComplexF64

    F(mu) = begin
        i, j = decode_pair(mu)
        (i > N || j > N) ? zero(T) : T(A[i, j])
    end
    return F, R, T
end

"""Reconstruye la matriz N×N COMPLETA evaluando el TT en los N² pares."""
# Dado un tensor train tt que representa A[i,j] codificado en quantics, reconstruye la matriz completa N×N evaluando el TT en cada par (i,j).
function reconstruct_matrix(tt, R::Int, N::Int, ::Type{T}) where {T}
    A = Matrix{T}(undef, N, N)
    for i in 1:N, j in 1:N
        A[i, j] = tt(encode_pair(i, j, R))
    end
    return A
end
# Esto recorre literalmente los N² pares (i,j), codifica cada uno en quantics, evalúa el tensor train,
# y pone el resultado en la matriz. Es fuerza bruta. Para N=32, son 1024 evaluaciones del MPO.
# Para N=64, son 4096.


# genera puntos iniciales "semilla" para que TCI arranque bien. TCI es un algoritmo iterativo que necesita pivotes iniciales; si empiezas en puntos malos, puede converger lento o mal.
function make_pivots(A::AbstractMatrix, R::Int; maxnz=2000, rng=MersenneTwister(1234), add_random=true)
    N = size(A, 1)
    seen = Set{Tuple{Vararg{Int}}}()
    pivs = Vector{Vector{Int}}()

    function add!(i, j)
        p = encode_pair(i, j, R)
        tp = Tuple(p)
        if tp ∉ seen
            push!(seen, tp)
            push!(pivs, p)
        end
    end

    # Primero agregamos los pivots diagonales (i,i) para asegurar que el MPO capture la parte diagonal de A, que suele ser importante.
    for i in 1:N
        add!(i, i)
    end

    # Luego agregamos pivots desde los no-nulos más grandes de A, para capturar la estructura más relevante.
    I, J, V = findnz(sparse(A))
    perm = sortperm(abs.(V), rev=true)
    for k in perm[1:min(maxnz, length(perm))]
        add!(I[k], J[k])
    end

    # Finalmente, opcionalmente agregamos algunos pivots aleatorios para mejorar la cobertura y evitar sesgos.
    if add_random
        for _ in 1:20
            add!(rand(rng, 1:N), rand(rng, 1:N))
        end
    end

    return pivs
end



# ═══════════════════════════════════════════════════════════════════════════════
# 3. GENERADORES DE HAMILTONIANOS — QUANTICA  (todos usan region → finitos)
# ═══════════════════════════════════════════════════════════════════════════════

function build_chain_1d(Nsites::Int; t::Real=1.0)
    h = LatticePresets.linear() |> hopping(t) |>
        supercell(region = r -> 0 <= r[1] < Nsites)
    Hs = sparse(h(()))
    (h, Hs, size(Hs, 1))
end

function build_square(; shape::Symbol, L::Real, t::Real=1.0)
    region = if shape == :square
        r -> abs(r[1]) < L && abs(r[2]) < L
    elseif shape == :disk
        r -> r[1]^2 + r[2]^2 < L^2
    elseif shape == :triangle
        r -> r[2] > -L/2 &&
             r[2] < sqrt(3)/2*L - L/2 &&
             r[2] < -sqrt(3)*r[1] + sqrt(3)/2*L - L/2 &&
             r[2] <  sqrt(3)*r[1] + sqrt(3)/2*L - L/2
    else error("shape: $shape") end
    h = LatticePresets.square() |> hopping(t) |> supercell(region=region)
    Hs = sparse(h(()))
    (h, Hs, size(Hs, 1))
end

function build_triangular(; shape::Symbol, L::Real, t::Real=1.0)
    region = if shape == :disk
        r -> r[1]^2 + r[2]^2 < L^2
    elseif shape == :hexagon
        r -> abs(r[1]) < L && abs(r[2]) < L*sqrt(3)/2 &&
             abs(r[2]) + sqrt(3)*abs(r[1]) < L*sqrt(3)
    else error("shape: $shape") end
    h = LatticePresets.triangular() |> hopping(t) |> supercell(region=region)
    Hs = sparse(h(()))
    (h, Hs, size(Hs, 1))
end

function build_honeycomb(; shape::Symbol, L::Real, t::Real=1.0,
                          arms::Int=5, amp::Real=0.25)
    region = if shape == :disk
        r -> r[1]^2 + r[2]^2 < L^2
    elseif shape == :hexagon
        r -> abs(r[1]) < L && abs(r[2]) < L*sqrt(3)/2 &&
             abs(r[2]) + sqrt(3)*abs(r[1]) < L*sqrt(3)
    elseif shape == :starfish
        r -> sqrt(r[1]^2 + r[2]^2) < L*(1 + amp*cos(arms*atan(r[2], r[1])))
    else error("shape: $shape") end
    h = LatticePresets.honeycomb() |> hopping(t) |> supercell(region=region)
    Hs = sparse(h(()))
    (h, Hs, size(Hs, 1))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TORUS (a mano — PBC sin problemas de Bloch phases)
# ═══════════════════════════════════════════════════════════════════════════════

function build_torus_1d(Nsites::Int; t::Real=1.0)
    N = Nsites; τ = Float64(t)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:N
        j = mod1(i + 1, N)
        push!(rows, i); push!(cols, j); push!(vals, -τ)
        push!(rows, j); push!(cols, i); push!(vals, -τ)
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    coords = [(cos(2π*i/N), sin(2π*i/N)) for i in 1:N]  # anillo
    (nothing, Hs, N, coords)
end

# ── Cadena 1D con hoppings oscilantes (tipo SSH/Peierls) ─────────────────────

"""
    build_chain_oscillating(N; t0=1.0, δt=0.5, period=4)

Cadena con hopping modulado: t_n = t0 + δt·cos(2πn/period).
Modela cadenas tipo SSH (period=2) o Peierls.
"""
function build_chain_oscillating(Nsites::Int; t0::Real=1.0, δt::Real=0.5, period::Int=4)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:Nsites-1
        t_i = t0 + δt * cos(2π * i / period)
        push!(rows, i); push!(cols, i+1); push!(vals, -t_i)
        push!(rows, i+1); push!(cols, i); push!(vals, -t_i)
    end
    Hs = sparse(rows, cols, vals, Nsites, Nsites); dropzeros!(Hs)
    coords = [(Float64(i), 0.0) for i in 1:Nsites]
    hoppings = [t0 + δt * cos(2π * i / period) for i in 1:Nsites-1]
    (nothing, Hs, Nsites, coords, hoppings)
end

# ── Cadena 1D con potencial quasiperiódico sin(√3·n) ────────────────────────

"""
    build_chain_quasiperiodic(N; t=1.0, V=1.0, β=sqrt(3))

Cadena con potencial on-site V·sin(β·n).
β irracional → potencial cuasiperiódico (no periódico nunca).
"""
function build_chain_quasiperiodic(Nsites::Int; t::Real=1.0, V::Real=1.0, β::Real=sqrt(3))
    rows = Int[]; cols = Int[]; vals = Float64[]
    # Hopping NN
    for i in 1:Nsites-1
        push!(rows, i); push!(cols, i+1); push!(vals, -t)
        push!(rows, i+1); push!(cols, i); push!(vals, -t)
    end
    # Potencial on-site
    for i in 1:Nsites
        push!(rows, i); push!(cols, i); push!(vals, V * sin(β * i))
    end
    Hs = sparse(rows, cols, vals, Nsites, Nsites); dropzeros!(Hs)
    coords = [(Float64(i), 0.0) for i in 1:Nsites]
    potential = [V * sin(β * i) for i in 1:Nsites]
    (nothing, Hs, Nsites, coords, potential)
end

# ── Modelo de Aubry-André ────────────────────────────────────────────────────

"""
    build_aubry_andre(N; t=1.0, V=1.0, β=(1+√5)/2, φ=0.0)

Modelo de Aubry-André:
  H = -t Σ (c†ᵢcᵢ₊₁ + h.c.) + V Σ cos(2πβn + φ) c†ₙcₙ

β = ratio áureo → potencial cuasiperiódico incomensurable.
Transición de localización en V = 2t:
  V < 2t → todos los estados extendidos
  V > 2t → todos los estados localizados
"""
function build_aubry_andre(Nsites::Int; t::Real=1.0, V::Real=1.0,
                           β::Real=(1+sqrt(5))/2, φ::Real=0.0)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:Nsites-1
        push!(rows, i); push!(cols, i+1); push!(vals, -t)
        push!(rows, i+1); push!(cols, i); push!(vals, -t)
    end
    for i in 1:Nsites
        push!(rows, i); push!(cols, i); push!(vals, V * cos(2π * β * i + φ))
    end
    Hs = sparse(rows, cols, vals, Nsites, Nsites); dropzeros!(Hs)
    coords = [(Float64(i), 0.0) for i in 1:Nsites]
    potential = [V * cos(2π * β * i + φ) for i in 1:Nsites]
    (nothing, Hs, Nsites, coords, potential)
end

# ── 2D cuadrada con potencial cuasiperiódico separable ───────────────────────

"""
    build_square_quasiper_separable(Lx, Ly; t, V, β)

Red cuadrada Lx×Ly con potencial on-site:
  V(nx, ny) = V · sin(β·nx) · sin(β·ny)
Potencial separable: producto de funciones 1D cuasiperiódicas.
"""
function build_square_quasiper_separable(Lx::Int, Ly::Int;
                                         t::Real=1.0, V::Real=1.5, β::Real=sqrt(3))
    N = Lx * Ly
    idx(x, y) = x + Lx * (y - 1)
    rows = Int[]; cols = Int[]; vals = Float64[]
    # Hoppings NN
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        if x < Lx; j = idx(x+1, y); push!(rows,i); push!(cols,j); push!(vals,-t)
                    push!(rows,j); push!(cols,i); push!(vals,-t); end
        if y < Ly; j = idx(x, y+1); push!(rows,i); push!(cols,j); push!(vals,-t)
                    push!(rows,j); push!(cols,i); push!(vals,-t); end
    end
    # Potencial on-site separable
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        push!(rows, i); push!(cols, i); push!(vals, V * sin(β * x) * sin(β * y))
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    coords = [(Float64(x), Float64(y)) for x in 1:Lx for y in 1:Ly]
    pot = [V * sin(β * x) * sin(β * y) for x in 1:Lx for y in 1:Ly]
    (nothing, Hs, N, coords, pot)
end

# ── 2D cuadrada con potencial cuasiperiódico no-separable ────────────────────

"""
    build_square_quasiper_coupled(Lx, Ly; t, V, β)

Red cuadrada Lx×Ly con potencial on-site:
  V(nx, ny) = V · sin(β·(nx + ny))
Potencial no-separable: mezcla las dos direcciones.
Más difícil de comprimir que el separable.
"""
function build_square_quasiper_coupled(Lx::Int, Ly::Int;
                                       t::Real=1.0, V::Real=1.5, β::Real=sqrt(3))
    N = Lx * Ly
    idx(x, y) = x + Lx * (y - 1)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        if x < Lx; j = idx(x+1, y); push!(rows,i); push!(cols,j); push!(vals,-t)
                    push!(rows,j); push!(cols,i); push!(vals,-t); end
        if y < Ly; j = idx(x, y+1); push!(rows,i); push!(cols,j); push!(vals,-t)
                    push!(rows,j); push!(cols,i); push!(vals,-t); end
    end
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        push!(rows, i); push!(cols, i); push!(vals, V * sin(β * (x + y)))
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    coords = [(Float64(x), Float64(y)) for x in 1:Lx for y in 1:Ly]
    pot = [V * sin(β * (x + y)) for x in 1:Lx for y in 1:Ly]
    (nothing, Hs, N, coords, pot)
end

function build_torus_square(Lx::Int, Ly::Int; t::Real=1.0)
    N = Lx * Ly; τ = Float64(t)
    idx(x, y) = mod1(x, Lx) + Lx*(mod1(y, Ly) - 1)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        for (dx, dy) in [(1,0), (0,1)]
            j = idx(x+dx, y+dy)
            push!(rows, i); push!(cols, j); push!(vals, -τ)
            push!(rows, j); push!(cols, i); push!(vals, -τ)
        end
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    coords = [(Float64(x), Float64(y)) for x in 1:Lx for y in 1:Ly]
    (nothing, Hs, N, coords)
end

function build_torus_triangular(Lx::Int, Ly::Int; t::Real=1.0)
    N = Lx * Ly; τ = Float64(t)
    idx(x, y) = mod1(x, Lx) + Lx*(mod1(y, Ly) - 1)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for x in 1:Lx, y in 1:Ly
        i = idx(x, y)
        for (dx, dy) in [(1,0), (0,1), (1,1)]
            j = idx(x+dx, y+dy)
            push!(rows, i); push!(cols, j); push!(vals, -τ)
            push!(rows, j); push!(cols, i); push!(vals, -τ)
        end
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    coords = [(x + 0.5*(y-1), y*sqrt(3)/2) for x in 1:Lx for y in 1:Ly]
    (nothing, Hs, N, coords)
end

function build_torus_honeycomb(Lx::Int, Ly::Int; t::Real=1.0)
    N = 2*Lx*Ly; τ = Float64(t)
    idx(x, y, s) = 2*(mod1(x,Lx)-1 + Lx*(mod1(y,Ly)-1)) + s
    rows = Int[]; cols = Int[]; vals = Float64[]
    for x in 1:Lx, y in 1:Ly
        a = idx(x, y, 1)
        for (dx, dy) in [(0,0), (-1,0), (0,-1)]
            b = idx(x+dx, y+dy, 2)
            push!(rows, a); push!(cols, b); push!(vals, -τ)
            push!(rows, b); push!(cols, a); push!(vals, -τ)
        end
    end
    Hs = sparse(rows, cols, vals, N, N); dropzeros!(Hs)
    cA = [(x + 0.5*(y-1),       y*sqrt(3)/2           ) for x in 1:Lx for y in 1:Ly]
    cB = [(x + 0.5*(y-1) + 0.5, y*sqrt(3)/2 + sqrt(3)/6) for x in 1:Lx for y in 1:Ly]
    (nothing, Hs, N, vcat(cA, cB))
end



# ═══════════════════════════════════════════════════════════════════════════════
# 5. COORDENADAS Y ENLACES PARA VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# extrae las coordenadas (xs, ys) de un sistema sys, ya sea a partir de sys.h (Quantica) o de sys.coords (matriz explícita).
function get_coords(sys)
    if sys.h !== nothing
        lat = Quantica.lattice(sys.h)
        pts = collect(Quantica.sites(lat))

        isempty(pts) && return (Float64[], Float64[])

        xs = [Float64(p[1]) for p in pts]
        ys = [length(p) >= 2 ? Float64(p[2]) : 0.0 for p in pts]
        return (xs, ys)

    elseif haskey(sys, :coords) && sys.coords !== nothing
        return ([c[1] for c in sys.coords], [c[2] for c in sys.coords])
    end

    return (Float64[], Float64[])
end

# genera la lista de segmentos (enlaces/bonds) para visualización, a partir de la matriz de Hamiltoniano sys.Hs y las coordenadas sys.coords. Solo incluye enlaces donde Hs[i,j] es no-nulo.
function get_bonds(sys)
    xs, ys = get_coords(sys)
    isempty(xs) && return Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    N = size(sys.Hs, 1)
    segs = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    for j in 1:N, i in (j+1):N
        if sys.Hs[i,j] != 0 && i <= length(xs) && j <= length(xs)
            push!(segs, ((xs[i],ys[i]), (xs[j],ys[j])))
        end
    end
    segs
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

"""
    full_pipeline(A, name; tol, maxiter)

Ejecuta el pipeline completo:
  1. Codificación quantics (d=4)
  2. Pivots globales inciales (diagonal + mayores no-nulos + aleatorios)
  3. crossinterpolate2 (2-site TCI)
  4. Reconstrucción COMPLETA N×N
  5. Validación: err∞, errFrob, hermiticidad, espectro
"""
function full_pipeline(A::AbstractMatrix, name::String;
                       tol::Real=DEFAULT_TOL,
                       maxiter::Int=DEFAULT_MAXITER,
                       rng=MersenneTwister(1234),
                       audit_tol::Real=1e-10)

    N = size(A, 1)
    if maxiter == 0
        maxiter = N > 100 ? 30 : DEFAULT_MAXITER
    end

    F, R, T = make_oracle(A)
    pivots = make_pivots(A, R; rng=rng)

    # Ejecutamos TCI con los pivots globales. TCI es un algoritmo iterativo que va ajustando el TT para que coincida con F en los pivots, y luego evalúa el error en otros puntos para decidir cómo ajustar el TT.
    # TCI.crossinterpolate2 ejecuta el algoritmo TCI de 2 sitios
    # devuelve el tensor train (MPO) que aproxima la matriz A, y también el error en los pivots (que debería ser pequeño si TCI convergió bien).
    tt, _, _ = TCI.crossinterpolate2(
        T, F, fill(4, R),
        [TCI.MultiIndex(p) for p in pivots];
        tolerance=tol,
        maxiter=maxiter
    )

    # reconstruye la matriz completa A_rec a partir del TT, evaluando el TT en cada par (i,j) codificado en quantics. Esto es fuerza bruta, pero para N=32 es factible.
    A_rec = reconstruct_matrix(tt, R, N, T)

    # Validamos la reconstrucción comparando A_rec con A. Calculamos el error máximo (norma ∞), el error de Frobenius, el error de hermiticidad, y la diferencia en los espectros (autovalores).
    # También identificamos el peor punto (i,j) donde la reconstrucción falla más.
    v = validate_reconstruction(Matrix(A), A_rec)

    ev_ex = sort(real.(eigvals(Matrix(A))))
    ev_re = sort(real.(eigvals(A_rec)))
    errλ  = maximum(abs.(ev_ex .- ev_re))

    ranks = [size(tt.sitetensors[ℓ], 3) for ℓ in 1:R-1]
    χ     = maximum(ranks)

    ok = v.err∞ < audit_tol && v.errH < audit_tol

    @printf("%-28s N=%4d R=%d χ=%3d  err∞=%.1e  errF=%.1e  errH=%.1e  errλ=%.1e %s\n",
            name, N, R, χ, v.err∞, v.errF, v.errH, errλ, ok ? "✓" : "✗")

    return (
        name=name, N=N, R=R, chi=χ, ranks=ranks,
        err∞=v.err∞, errF=v.errF, errH=v.errH, errλ=errλ,
        worst_index=v.worst_index,
        worst_exact=v.worst_exact,
        worst_rec=v.worst_rec,
        worst_rel=v.worst_rel,
        top_errors=v.top_errors,
        tt=tt, A_rec=A_rec, A_orig=Matrix(A), ok=ok
    )
end


function validate_reconstruction(A, A_rec; topk::Int=10)
    Δ = A .- A_rec
    absΔ = abs.(Δ)

    err∞, idx = findmax(absΔ)
    iworst, jworst = Tuple(idx)

    denom = max(norm(A), eps(real(float(one(eltype(A))))))
    errF = norm(Δ) / denom
    errH = maximum(abs.(A_rec .- A_rec'))

    inds = sortperm(vec(absΔ), rev=true)[1:min(topk, length(absΔ))]
    nr, nc = size(A)

    top_errors = NamedTuple[]
    for ind in inds
        i = ((ind - 1) % nr) + 1
        j = ((ind - 1) ÷ nr) + 1
        ex = A[i, j]
        re = A_rec[i, j]
        ab = abs(ex - re)
        rl = ab / max(abs(ex), eps(real(float(one(eltype(A))))))
        push!(top_errors, (i=i, j=j, exact=ex, rec=re, abs_err=ab, rel_err=rl))
    end

    worst_rel = err∞ / max(abs(A[iworst, jworst]), eps(real(float(one(eltype(A))))))

    return (
        err∞ = err∞,
        errF = errF,
        errH = errH,
        worst_index = (iworst, jworst),
        worst_exact = A[iworst, jworst],
        worst_rec   = A_rec[iworst, jworst],
        worst_rel   = worst_rel,
        top_errors  = top_errors,
    )
end

function extract_cores(tt)
    if hasproperty(tt, :sitetensors)
        return getproperty(tt, :sitetensors)
    elseif :sitetensors in fieldnames(typeof(tt))
        return getfield(tt, :sitetensors)
    else
        error("No encuentro los cores del TT/MPO. Mira fieldnames(typeof(tt)).")
    end
end

function core_dimensions(cores)
    [(size(G,1), size(G,2), size(G,3)) for G in cores]
end

function quantics_table(N::Int, R::Int)
    rows = NamedTuple[]
    for i in 1:N, j in 1:N
        mu = encode_pair(i, j, R)
        push!(rows, (i=i, j=j, mu=mu))
    end
    rows
end

function pivot_table(pivots)
    rows = NamedTuple[]
    for p in pivots
        i, j = decode_pair(p)
        push!(rows, (i=i, j=j, mu=p))
    end
    rows
end

function mpo_entry_from_cores(cores, mu)
    T0 = eltype(cores[1])
    M = ones(T0, 1, 1)
    for ℓ in 1:length(cores)
        M = M * cores[ℓ][:, mu[ℓ], :]
    end
    return M[1,1]
end

function mpo_full_report(A::AbstractMatrix, name::String;
                         tol::Real=DEFAULT_TOL,
                         maxiter::Int=DEFAULT_MAXITER,
                         rng=MersenneTwister(1234),
                         topk::Int=10)

    N = size(A,1)
    F, R, T = make_oracle(A)
    pivots = make_pivots(A, R; rng=rng)

    tt, _, _ = TCI.crossinterpolate2(
        T, F, fill(4, R),
        [TCI.MultiIndex(p) for p in pivots];
        tolerance=tol,
        maxiter=maxiter)

    A_rec = reconstruct_matrix(tt, R, N, T)
    v = validate_reconstruction(Matrix(A), A_rec; topk=topk)

    cores = extract_cores(tt)
    dims  = core_dimensions(cores)
    ranks = [size(tt.sitetensors[ℓ], 3) for ℓ in 1:length(tt.sitetensors)-1]

    return (
        name = name,
        N = N,
        R = R,
        T = T,
        A = Matrix(A),
        F = F,
        pivots = pivots,
        pivots_table = pivot_table(pivots),
        quantics = quantics_table(N, R),
        tt = tt,
        cores = cores,
        core_dims = dims,
        ranks = ranks,
        chi = maximum(ranks),
        A_rec = A_rec,
        validation = v
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CATÁLOGO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

function build_all_systems()
    S = []
    # 1D (N=32 todos)
    let (h, H, N) = build_chain_1d(32)
        push!(S, (name="Cadena 1D abierta", h=h, Hs=H, N=N, coords=nothing, cat="1D"))
    end
    let (_, H, N, c) = build_torus_1d(32)
        push!(S, (name="Torus 1D (anillo)", h=nothing, Hs=H, N=N, coords=c, cat="1D"))
    end
    # 1D con hoppings oscilantes
    let (_, H, N, c, _) = build_chain_oscillating(32; t0=1.0, δt=0.5, period=4)
        push!(S, (name="Cadena oscilante", h=nothing, Hs=H, N=N, coords=c, cat="1D"))
    end
    # 1D quasiperiódico sin(√3·n)
    let (_, H, N, c, _) = build_chain_quasiperiodic(32; V=1.5, β=sqrt(3))
        push!(S, (name="Cadena sin(√3·n)", h=nothing, Hs=H, N=N, coords=c, cat="1D"))
    end
    # Aubry-André (fase extendida V<2t)
    let (_, H, N, c, _) = build_aubry_andre(32; V=1.0)
        push!(S, (name="Aubry-André V=1", h=nothing, Hs=H, N=N, coords=c, cat="1D"))
    end
    # Aubry-André (fase localizada V>2t)
   # let (_, H, N, c, _) = build_aubry_andre(32; V=3.0)
    #    push!(S, (name="Aubry-André V=3", h=nothing, Hs=H, N=N, coords=c, cat="1D"))
    #end

    # 2D cuasiperiódica separable sin(√3·nx)·sin(√3·ny)
    let (_, H, N, c, _) = build_square_quasiper_separable(6, 6; V=1.5)
        push!(S, (name="2D sin·sin separable", h=nothing, Hs=H, N=N, coords=c, cat="Cuadrada"))
    end
    # 2D cuasiperiódica acoplada sin(√3·(nx+ny))
    let (_, H, N, c, _) = build_square_quasiper_coupled(6, 6; V=1.5)
        push!(S, (name="2D sin acoplada", h=nothing, Hs=H, N=N, coords=c, cat="Cuadrada"))
    end

    # Cuadrada (L ajustado para N~25-49)
    for (s, l, sz) in [(:square,"Cuadrado",4.0), (:disk,"Disco",3.5), (:triangle,"Triángulo",9.0)]
        h, H, N = build_square(shape=s, L=sz)
        push!(S, (name="Cuad. $l", h=h, Hs=H, N=N, coords=nothing, cat="Cuadrada"))
    end
    # Triangular (L ajustado para N~30-40)
    for (s, l) in [(:disk,"Disco"), (:hexagon,"Hexágono")]
        h, H, N = build_triangular(shape=s, L=3.5)
        push!(S, (name="Triang. $l", h=h, Hs=H, N=N, coords=nothing, cat="Triangular"))
    end
    # Honeycomb (L pequeño para N~40-70)
    for (s, l, sz) in [(:disk,"Disco",3.0), (:hexagon,"Hexágono",3.0), (:starfish,"Starfish",3.0)]
        h, H, N = build_honeycomb(shape=s, L=sz)
        push!(S, (name="Grafeno $l", h=h, Hs=H, N=N, coords=nothing, cat="Honeycomb"))
    end
    # Torus 2D (L=4 para N~16-32)
    let L = 5
        _, H, N, c = build_torus_square(L, L)
        push!(S, (name="Torus Cuadrado", h=nothing, Hs=H, N=N, coords=c, cat="Torus"))
        _, H, N, c = build_torus_triangular(L, L)
        push!(S, (name="Torus Triangular", h=nothing, Hs=H, N=N, coords=c, cat="Torus"))
        _, H, N, c = build_torus_honeycomb(L, L)
        push!(S, (name="Torus Honeycomb", h=nothing, Hs=H, N=N, coords=c, cat="Torus"))
    end
    S
end

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ESTUDIO DE ESCALADO  χ_max vs N
# ═══════════════════════════════════════════════════════════════════════════════

"""
    scaling_study(builder, sizes, label; tol, maxiter)

Para una geometría dada, varía el parámetro de tamaño y registra cómo
crece χ_max con N. Solo valida para tamaños donde N ≤ maxN_validate.

builder(sz) debe retornar (h_or_nothing, Hs, N, ...)
"""
function scaling_study(builder, sizes, label::String;
                       tol::Real=DEFAULT_TOL,
                       maxiter::Int=DEFAULT_MAXITER,
                       maxN_validate::Int=4000,
                       rng=MersenneTwister(1234),
                       numpivots = 10)

    results = NamedTuple[]

    for sz in sizes
        out = builder(sz)
        Hs = out[2]
        N  = out[3]
        N < 4 && continue

        F, R, T = make_oracle(Hs)
        #pivots = make_pivots(Hs, R; rng=rng)
        rows, cols, vals = findnz(Hs)
        pivots = [encode_pair(rows[n], cols[n], R) for n in 1:min(length(vals), numpivots)]

        mi = N > 200 ? 100 : N > 100 ? 30 : maxiter

        tt, _, _ = TCI.crossinterpolate2(
            T, F, fill(4, R),
            [TCI.MultiIndex(p) for p in pivots];
            tolerance=tol,
            maxiter=mi,
        )

        ranks_s = [size(tt.sitetensors[ℓ], 3) for ℓ in 1:length(tt.sitetensors)-1]
        χ = maximum(ranks_s)

        err∞ = NaN
        errH = NaN
        if N <= maxN_validate
            A_rec = reconstruct_matrix(tt, R, N, T)
            v = validate_reconstruction(Matrix(Hs), A_rec)
            err∞ = v.err∞
            errH = v.errH
        end

        push!(results, (size=sz, N=N, R=R, chi=χ, err=err∞, errH=errH, nnz=nnz(Hs)))

        @printf("  %s  sz=%5.1f  N=%4d  R=%d  χ=%3d  err∞=%.1e  errH=%.1e\n",
                label, Float64(sz), N, R, χ, err∞, errH)
    end

    return results
end




# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALIZACIÓN DE SISTEMAS
# ═══════════════════════════════════════════════════════════════════════════════

function plot_system(sys)
    Hd = real.(Matrix(sys.Hs))
    N  = sys.N
    xs, ys = get_coords(sys)
    segs   = get_bonds(sys)

    # ── Panel izquierdo: red ──
    if sys.cat == "1D" && !isempty(xs)
        diag_vals = [Hd[i,i] for i in 1:N]
        has_pot   = maximum(abs.(diag_vals)) > 1e-10
        hop_vals  = [abs(Hd[i, i+1]) for i in 1:N-1]
        hop_var   = maximum(hop_vals) - minimum(hop_vals) > 1e-10

        if has_pot
            # ── Potencial on-site: sitios a la altura V(n) ──
            p1 = plot(title="$(sys.name) (N=$N)",
                      xlabel="sitio n", ylabel="potencial V(n)", titlefontsize=10)
            for i in 1:N-1
                plot!(p1, [i, i+1], [diag_vals[i], diag_vals[i+1]],
                      color=:gray70, lw=1.5, label=false)
            end
            scatter!(p1, 1:N, diag_vals, ms=5, color=:steelblue,
                     markerstrokewidth=0.5, label="V(n)")
            hline!(p1, [0], color=:black, ls=:dash, alpha=0.3, label=false)

        elseif hop_var
            # Cadena con hoppings variables: mostrar valor de cada hopping
            p1 = plot(title="$(sys.name) (N=$N)",
                      xlabel="enlace n→n+1", ylabel="|t(n)|",
                      titlefontsize=10, legend=false)
            bar!(p1, 1:N-1, hop_vals, color=:steelblue, bar_width=0.7, alpha=0.8)

        else
            # ── Cadena/torus uniforme ──
            p1 = plot(title="$(sys.name) (N=$N)",
                      xlabel="sitio n", titlefontsize=10, legend=false)
            if occursin("Torus", sys.name) || occursin("anillo", sys.name)
                # Dibujar como ANILLO real
                θs = range(0, 2π, length=N+1)[1:N]
                cx = cos.(θs); cy = sin.(θs)
                for i in 1:N
                    j = mod1(i+1, N)
                    plot!(p1, [cx[i], cx[j]], [cy[i], cy[j]], color=:gray, lw=2, label=false)
                end
                scatter!(p1, cx, cy, ms=5, color=:navy, markerstrokewidth=0, label=false,
                         aspect_ratio=:equal, grid=false)
            else
                plot!(p1, 1:N, zeros(N), color=:gray, lw=2, label=false)
                scatter!(p1, 1:N, zeros(N), ms=4, color=:navy, markerstrokewidth=0, label=false)
            end
        end

    elseif !isempty(xs)
        # ════ 2D ════
        diag_vals = [i <= N ? Hd[i,i] : 0.0 for i in 1:length(xs)]
        has_pot = maximum(abs.(diag_vals)) > 1e-10

        xmin, xmax = minimum(xs)-0.5, maximum(xs)+0.5
        ymin, ymax = minimum(ys)-0.5, maximum(ys)+0.5
        p1 = plot(title="$(sys.name) (N=$N)",
                  aspect_ratio=:equal, legend=false, grid=false, titlefontsize=10,
                  xlims=(xmin, xmax), ylims=(ymin, ymax))
        for ((x1,y1),(x2,y2)) in segs
            plot!(p1, [x1,x2], [y1,y2], color=:gray80, lw=1)
        end
        if has_pot
            scatter!(p1, xs, ys, zcolor=diag_vals[1:length(xs)], color=:RdBu,
                     ms=5, markerstrokewidth=0, colorbar=true, colorbar_title="V(n)")
        else
            scatter!(p1, xs, ys, ms=5, color=:navy, markerstrokewidth=0)
        end


    else
        # Sin coordenadas — solo título
        p1 = plot(title="$(sys.name) (N=$N) — sin coords",
                  titlefontsize=10, legend=false, grid=false)
    end

    # ── Panel derecho: ESTRUCTURA SPARSE (blanco/negro) ──
    # Negro = elemento no-nulo, blanco = cero
    spy_matrix = Float64.(Hd .== 0)
    p2 = heatmap(spy_matrix, color=:grays, yflip=true, aspect_ratio=:equal,
                 title="Estructura sparse H ($N×$N)",
                 titlefontsize=9, tickfontsize=7, colorbar=false)

    plot(p1, p2, layout=(1,2), size=(800, 350), top_margin=3Plots.mm)

end


function plot_detail(res)
    N = res.N
    R = length(res.ranks)

    # Panel 1: error elemento a elemento
    p1 = heatmap(log10.(abs.(real.(res.A_orig) .- real.(res.A_rec)) .+ 1e-16),
                 color=:viridis, clims=(-16, 0), yflip=true, aspect_ratio=:equal,
                 title="log₁₀|H − H_MPO|", titlefontsize=9, colorbar_title="log₁₀")

    # Panel 2: espectro — AMBOS siempre visibles
    ev_ex = sort(real.(eigvals(res.A_orig)))
    ev_re = sort(real.(eigvals(res.A_rec)))
    p2 = scatter(1:N, ev_re, label="MPO", ms=5, color=:red,    # ← MPO primero, grande
                 marker=:x, markerstrokewidth=2,
                 title="Espectro", xlabel="índice", ylabel="E",
                 titlefontsize=9, legendfontsize=7)
    scatter!(p2, 1:N, ev_ex, label="Exacto", ms=3, color=:blue, # ← Exacto encima, pequeño
             marker=:circle, alpha=0.7)

    # Panel 3: bond dimensions — siempre visibles con valores encima
    p3 = bar(1:R, res.ranks, label=false, color=:steelblue,
             bar_width=0.7,
             ylims=(0, max(maximum(res.ranks) * 1.3, 3)),
             xlims=(0.3, R + 0.7),
             title="χ_ℓ  (χ_max=$(res.chi), R=$R)",
             xlabel="sitio ℓ", ylabel="χ_ℓ",
             titlefontsize=9, tickfontsize=8)
    for (i, v) in enumerate(res.ranks)
        annotate!(p3, i, v + max(maximum(res.ranks) * 0.08, 0.3),
                  text("$v", 8, :center, :bold))
    end

    plot(p1, p2, p3, layout=(1,3), size=(1000, 320),
         plot_title="$(res.name) — N=$(N), err∞=$(@sprintf("%.1e", res.err∞))",
         plot_titlefontsize=10, top_margin=5Plots.mm)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PIVOTS POR MODO — para exploración interactiva en el notebook
# ═══════════════════════════════════════════════════════════════════════════════

"""
    make_pivots_mode(A, R, mode; manual_pairs=[], rng=MersenneTwister(1234))

Genera pivotes según el modo elegido. Modos disponibles:
  "auto"              diagonal + no-nulos + aleatorios (referencia)
  "diagonal"          solo (i,i)
  "manual"            pares de manual_pairs
  "minimal"           5 aleatorios (casi seguro falla)
  "all_nonzeros"      todos los H[i,j]≠0
  "random_nonzeros"   10 no-nulos al azar
  "rowcover"          1 no-nulo por fila (cobertura mínima)
  "diagonal_plus_few" diagonal + 5 no-nulos más grandes
"""
function make_pivots_mode(A::AbstractMatrix, R::Int, mode::String;
                          manual_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                          rng=MersenneTwister(1234))
    N = size(A, 1)

    if mode == "auto"
        return make_pivots(sparse(A), R; rng=rng)

    elseif mode == "diagonal"
        return [encode_pair(i, i, R) for i in 1:N]

    elseif mode == "minimal"
        return [encode_pair(rand(rng, 1:N), rand(rng, 1:N), R) for _ in 1:5]

    elseif mode == "manual"
        isempty(manual_pairs) && error("modo manual requiere manual_pairs no vacío")
        return [encode_pair(i, j, R) for (i,j) in manual_pairs]

    elseif mode == "all_nonzeros"
        I, J, _ = findnz(sparse(A))
        return [encode_pair(I[k], J[k], R) for k in 1:length(I)]

    elseif mode == "random_nonzeros"
        I, J, _ = findnz(sparse(A))
        n_sample = min(10, length(I))
        idx = randperm(rng, length(I))[1:n_sample]
        return [encode_pair(I[k], J[k], R) for k in idx]

    elseif mode == "rowcover"
        pivs = Vector{Vector{Int}}()
        seen_rows = Set{Int}()
        I, J, V = findnz(sparse(A))
        for k in sortperm(abs.(V), rev=true)
            if I[k] ∉ seen_rows
                push!(seen_rows, I[k])
                push!(pivs, encode_pair(I[k], J[k], R))
            end
            length(seen_rows) >= N && break
        end
        return pivs

    elseif mode == "diagonal_plus_few"
        pivs = [encode_pair(i, i, R) for i in 1:N]
        I, J, V = findnz(sparse(A))
        offdiag = [(I[k], J[k], abs(V[k])) for k in 1:length(I) if I[k] != J[k]]
        sort!(offdiag, by=x->x[3], rev=true)
        for k in 1:min(5, length(offdiag))
            push!(pivs, encode_pair(offdiag[k][1], offdiag[k][2], R))
        end
        return pivs

    else
        error("Modo de pivots desconocido: $mode")
    end
end

"""
    build_custom_system(system_type, system_size)

Construye un sistema por nombre y tamaño. Devuelve (Hs_sparse, N, name).
"""
function build_custom_system(system_type::String, system_size)
    builders = Dict(
        "cadena"           => sz -> (build_chain_1d(Int(sz))...,                    "Cadena 1D N=$sz"),
        "torus1d"          => sz -> (build_torus_1d(Int(sz))...,                    "Torus 1D N=$sz"),
        "oscilante"        => sz -> (build_chain_oscillating(Int(sz))...,           "Oscilante N=$sz"),
        "sin_sqrt3"        => sz -> (build_chain_quasiperiodic(Int(sz))...,         "sin(√3·n) N=$sz"),
        "aubry_andre"      => sz -> (build_aubry_andre(Int(sz))...,                 "Aubry-André N=$sz"),
        "cuad_cuadrado"    => sz -> (build_square(shape=:square, L=sz)...,          "Cuad.Cuadrado L=$sz"),
        "cuad_disco"       => sz -> (build_square(shape=:disk, L=sz)...,            "Cuad.Disco L=$sz"),
        "cuad_triangulo"   => sz -> (build_square(shape=:triangle, L=sz)...,        "Cuad.Triángulo L=$sz"),
        "triang_disco"     => sz -> (build_triangular(shape=:disk, L=sz)...,        "Triang.Disco L=$sz"),
        "triang_hexagono"  => sz -> (build_triangular(shape=:hexagon, L=sz)...,     "Triang.Hex L=$sz"),
        "grafeno_disco"    => sz -> (build_honeycomb(shape=:disk, L=sz)...,         "Grafeno Disco L=$sz"),
        "grafeno_hexagono" => sz -> (build_honeycomb(shape=:hexagon, L=sz)...,      "Grafeno Hex L=$sz"),
        "grafeno_starfish" => sz -> (build_honeycomb(shape=:starfish, L=sz)...,     "Grafeno Star L=$sz"),
        "torus_cuadrado"   => sz -> (build_torus_square(Int(sz),Int(sz))...,        "Torus Cuad L=$sz"),
        "torus_triangular" => sz -> (build_torus_triangular(Int(sz),Int(sz))...,    "Torus Triang L=$sz"),
        "torus_honeycomb"  => sz -> (build_torus_honeycomb(Int(sz),Int(sz))...,     "Torus HC L=$sz"),
        "sin_sin_sep"      => sz -> (build_square_quasiper_separable(Int(sz),Int(sz))..., "2D sin·sin L=$sz"),
        "sin_acoplada"     => sz -> (build_square_quasiper_coupled(Int(sz),Int(sz))...,   "2D sin(n+n) L=$sz"),
    )
    haskey(builders, system_type) || error("Sistema desconocido: $system_type. Disponibles: $(sort(collect(keys(builders))))")
    out = builders[system_type](system_size)
    Hs = out[2]
    name = out[end]
    return (Hs=Hs, N=size(Hs, 1), name=name)
end

"""
    anatomy_tci(A_orig, name, pivots_used, R;
                tol=1e-10, maxiter=30, validate_max_N=64, sample_size=500)

Ejecuta TCI con los pivotes dados y devuelve un NamedTuple con todo:
  tt, A_rec, ranks, χ, errores, validación elemento a elemento, etc.
Devuelve nothing si TCI falla (pivotes en ceros).
"""
function anatomy_tci(A_orig::Matrix{Float64}, name::String,
                     pivots_used::Vector{Vector{Int}}, R::Int;
                     tol::Real=1e-10, maxiter::Int=30,
                     validate_max_N::Int=64, sample_size::Int=500)
    N = size(A_orig, 1)
    F, _, T = make_oracle(sparse(A_orig))

    tt, _, _ = TCI.crossinterpolate2(
        T, F, fill(4, R),
        [TCI.MultiIndex(p) for p in pivots_used];
        tolerance=tol, maxiter=maxiter)

    ranks = [size(tt.sitetensors[ℓ], 3) for ℓ in 1:R-1]
    χ = maximum(ranks)
    A_rec = Float64.(real.(reconstruct_matrix(tt, R, N, T)))

    err∞ = maximum(abs.(A_orig .- A_rec))
    errF = norm(A_orig .- A_rec) / max(norm(A_orig), 1e-300)
    ev_ex = sort(real.(eigvals(A_orig)))
    ev_re = sort(real.(eigvals(A_rec)))
    errλ = maximum(abs.(ev_ex .- ev_re))
    errH = maximum(abs.(A_rec .- A_rec'))

    # Validación elemento a elemento
    max_err = 0.0; sum_err = 0.0; n_wrong = 0
    worst_i, worst_j = 1, 1; n_nz = 0; n_checked = 0

    if N <= validate_max_N
        for i in 1:N, j in 1:N
            exact = A_orig[i, j]
            mpo_val = tt(encode_pair(i, j, R))
            err = abs(exact - mpo_val)
            sum_err += err; n_checked += 1
            if exact != 0; n_nz += 1; end
            if err > max_err; max_err = err; worst_i = i; worst_j = j; end
            if err > 1e-10; n_wrong += 1; end
        end
    else
        n_checked = sample_size
        rng = MersenneTwister(42)
        for _ in 1:sample_size
            i, j = rand(rng, 1:N), rand(rng, 1:N)
            err = abs(A_orig[i,j] - tt(encode_pair(i, j, R)))
            if err > max_err; max_err = err; worst_i = i; worst_j = j; end
        end
    end

    total_params = sum(prod(size(tt.sitetensors[ℓ])) for ℓ in 1:R)

    return (
        name=name, N=N, R=R, chi=χ, ranks=ranks,
        err∞=err∞, errF=errF, errH=errH, errλ=errλ,
        tt=tt, A_rec=A_rec, A_orig=A_orig,
        ok=err∞ < 1e-6,
        total_params=total_params,
        compression=N*N / total_params,
        validation=(max_err=max_err, sum_err=sum_err,
                    n_wrong=n_wrong, n_nz=n_nz, n_checked=n_checked,
                    worst=(worst_i, worst_j), sampled=N > validate_max_N),
    )
end
