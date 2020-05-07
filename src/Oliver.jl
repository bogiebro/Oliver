module Oliver
using LinearAlgebra
using Optim
import ConstructionBase: constructorof
using StaticArrays
using LinearMaps
using LinearMapDuals
using ToeplitzMatrices
using ForwardDiff

export cov, Kern, StationaryKern, Ksm, IxKern, approx_log_p, k_inv_y

# Make Flatten play nice with StaticArrays
constructorof(::Type{MArray{A,B,C,D}}) where {A,B,C,D} = MArray{A}
constructorof(::Type{SArray{A,B,C,D}}) where {A,B,C,D} = SArray{A}

abstract type Kern end

abstract type StationaryKern <: Kern end

struct Ksm{N, T<: Real} <: StationaryKern
    weights::SVector{N, T}
    sigmas::SVector{N, T}
    mus::SVector{N, T}
end
Ksm(n::Int) = Ksm{n, Float64}(randn(n), ones(n), randn(n))
kern(k::Ksm, tau) = sum(k.weights.^2 .* exp.(-2*pi^2 * tau^2 .* k.sigmas.^2) .* cos.(2*pi * tau .* k.mus))

# Note: points in a and b here must be sampled at the same interval
function cov(k::StationaryKern, a, b)
    kk = kern.(Ref(k),(a .- b[1]))
    LinearMap(SymmetricToeplitz(kk); issymmetric=true, isposdef=true)
end

# wait this is totally wrong
# we have to multiply the two lower triangular ones together. 
# Let's check out Kaledio
struct IxKern{N, T<:Real} <: Kern
    v::SVector{N,T}
end
IxKern(n::Int) = IxKern{n, Float64}(randn(2))
kern(k::IxKern, a,b) = (k.v * k.v')[a,b]

cov(k::Kern, a, b) = LinearMap(kern.(Ref(k),a, b'))
cov(k::Kern, a) = cov(k, a, a)

# TODO: cholesky preconditioner
# More sums can be in place
# we should test that this approximates true solving
function bbm(A, y, zs, t)::Tuple{Matrix{Float64}, Vector{SymTridiagonal{Float64}}}
    B = [y zs]
    X = zeros(size(B))
    R = mul!(similar(X), -A, X) + B # residuals
    res_len = sum(R .* R, dims=1)
    D = copy(R) # current directions
    V = mul!(similar(X), A, D)
    T = Array{Float64}(undef, t, t, 2) # tridiags by column
    alpha = ones(1, t+1) # error components A-parallel to current directions
    beta = zeros(1, t+1) # residual components A-parallel to previous directions
    for j in 1:t
        
        # if the residual is already tiny, we're done early
        if all(mapslices(norm, R, dims=1) .< 1e-10)
          return X, [SymTridiagonal(T[i,1:j-1,1], T[i,1:j-2,2]) for i in 1:t] 
        end
    
        prev_alpha = alpha
        alpha = res_len ./ sum(D .* V, dims=1)
        X .+= alpha .* D # remove the error comonents A-parallel to D
        R .-= alpha .* V # calculate new residual
        prev_res_len = res_len
        res_len = sum(R .* R, dims=1)  
        prev_beta = beta
        beta = res_len ./ prev_res_len # find A-parallel component of R wrt D
        D .= R .+ beta .* D # make the new direction conjugate to the previous
        mul!(V, A, D)
        T[:, j, 1] .= (1 ./ alpha .+ prev_beta ./ prev_alpha)[1,2:end]
        T[:,j, 2] .= (sqrt.(beta) ./ alpha)[1, 2:end]
    end
    X, [SymTridiagonal(T[i,:,1], T[i,1:end-1,2]) for i in 1:t] 
end

function approx_log_p(kxx::LinearMapDual{T,V,N}, y; C=I, t=100) where {T,V,N}
    t = min(t, length(y))
    z = randn(size(y, 1), t)
    k_inv_yz, ts = bbm(kxx.val, y, z, t)
    k_inv_y = C * k_inv_yz[:, 1]
    jacobians = Vector{Float64}(undef, N)
    for (i, d_kxx_i) in enumerate(kxx.jacobians)
        approx_tr = sum(k_inv_yz[:, 2:end] .* mul!(similar(z), d_kxx_i, z)) / t
        jacobians[i] = k_inv_y' * (d_kxx_i * k_inv_y) + approx_tr
    end
    logdet = 0.0
    for i in 1:t
      F = eigen(ts[i])
      logvec = F.vectors[1,:]
      logdet += logvec' * (Diagonal(log.(F.values)) * logvec)
    end
    logdet /= t
    ForwardDiff.Dual{T,V,N}(logdet - y' * k_inv_y, ForwardDiff.Partials(Tuple(jacobians)))
end

function k_inv_y(likelihood, x, y, model)
    kxx, C = likelihood(model, x)
    C * bbm(kxx, y, [])[1]
end

end # module
