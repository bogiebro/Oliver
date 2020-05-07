using Revise
using ReTester
using Tests
entr(run_tests(Tests.test), ["src/Oliver.jl", "Tests/src/Tests.jl"]; pause=0.2)
