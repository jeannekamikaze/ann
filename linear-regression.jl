using AutoGrad
using Knet

include(Knet.dir("data", "housing.jl"))

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = mean(abs2, y - predict(w,x))

lossgradient = grad(loss)

function train(w, data; lr=0.1)
    for (x,y) in data
        dw = lossgradient(w,x,y)
        for i = 1:length(w) # 0 -> weights, 1 -> bias
            w[i] = w[i] - lr*dw[i]
        end
    end
    return w
end

x,y = housing()
w = Any[0.1 * randn(1,13), 0.0]
for i = 1:10
    train(w, [(x,y)])
    println(loss(w,x,y))
end
