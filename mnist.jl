using AutoGrad
using Knet

include(Knet.dir("data", "mnist.jl"))

predict(w,x) = w[1]*mat(x) .+ w[2]

loss(w,x,y) = nll(predict(w,x), y)

lossgradient = grad(loss)

function train(w, data; lr=0.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        w[1] = w[1] - lr*dw[1]
        w[2] = w[2] - lr*dw[2]
    end
    return w
end

x_train, y_train, x_test, y_test = mnist()
d_train = minibatch(x_train, y_train, 100, shuffle=true)
d_test = minibatch(x_test, y_test, 100)

w = Any[0.1 * randn(10, 28^2), zeros(10,1)]

for epoch=1:15
    train(w, d_train)
    println(
        "epoch ", epoch,
        " train accuracy ", accuracy(x -> predict(w,x), d_train),
        " test accuracy ", accuracy(x -> predict(w,x), d_test))
end
