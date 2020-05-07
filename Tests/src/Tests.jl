module Tests
using Test
using Oliver
using Optim
using Flatten
using ForwardDiff
using DiffResults
using Debugger

function test()
    model = Ksm(4)
    x = collect(1.0:25.0)
    y = randn(25)
    params = collect(flatten(model))
    log_p(p)= -approx_log_p(cov(reconstruct(model, p), x), y)

    @testset "derivative is non-nan" begin
        result = DiffResults.GradientResult(params)
        result = ForwardDiff.gradient!(result, log_p, params)
        @test !any(isnan.(DiffResults.value(result)))
        @test all(isfinite.(DiffResults.value(result)))
        @test !any(isnan.(DiffResults.gradient(result)))
        @test all(isfinite.(DiffResults.gradient(result)))
    end

    L = randn(5,5)
    A = L * L'
    y_small = randn(5)
    z = randn(5, 5)

    @testset "bmm matches gaussian elim" begin
      bbm_sol = Oliver.bbm(A, y_small, z, 5)[1][:, 1]
      true_sol = A \ y_small
      @test bbm_sol ≈ true_sol
    end

    @testset "optim converges" begin
        results = optimize(log_p, collect(flatten(model)), GradientDescent(),
            Optim.Options(f_tol=1e-8, show_trace=true), autodiff=:forward)
        @test all(isfinite.(Optim.minimizer(results)))
    end

end

end # module
