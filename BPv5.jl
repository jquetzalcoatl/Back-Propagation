using Random, MLDatasets, Images, LinearAlgebra, PlotlyJS, DelimitedFiles
#include("Pickle.jl") GZip,

mutable struct network
    sizes
    biases
    weights
    numOfLayers
end

function networkF(args, initial_weight_type)
    n = network(args, [randn(y,1) for y in args[2:length(args)]],
    [weight_init(initial_weight_type, y, x) for (x, y) in zip(args[1:length(args)-1], args[2:length(args)]) ], length(args))    #divide weights by number of inputs (x)
end

function weight_init(initial_weight_type, y, x)
    if initial_weight_type == "Default Weight"
        return randn(y,x)/sqrt(x)
    elseif initial_weight_type == "Large Weight"
        return randn(y,x)
    end
end

function SGD(netw, epoch, minibatchsize, eta, cf, lambda, tr_x, tr_y, te_x, te_y, ev_x, ev_y)
    # tr_x, tr_y, te_x, te_y, ev_x, ev_y = loadData(size(netw.weights[2])[1], trainSize)
    println("Training...")
    println("epoch: 0 -- Accuracy on evaluation data: ", evaluateRes(netw, ev_x, ev_y))
    acc_tr = evaluateRes(netw, tr_x, tr_y, netw.weights)
    acc_ev = evaluateRes(netw, ev_x, ev_y, netw.weights)
    acc_te = evaluateRes(netw, te_x, te_y, netw.weights)
    cost_tr = [total_cost(netw, tr_x, tr_y, cf, lambda)]
    cost_ev = [total_cost(netw, ev_x, ev_y, cf, lambda)]
    cost_te = [total_cost(netw, te_x, te_y, cf, lambda)]

    for i=1:epoch
        # println("epoch: ", i)
        index = shuffle(collect(1:size(tr_x)[2]))
        tr_x = tr_x[:,index]
        tr_y = tr_y[:,index]
        mini_batch_x = [tr_x[:,k:k+minibatchsize] for k = 1:minibatchsize:size(tr_x)[2]-minibatchsize]
        mini_batch_y = [tr_y[:,k:k+minibatchsize] for k = 1:minibatchsize:size(tr_y)[2]-minibatchsize]
        #UPDATE NETWORK
        oldweights = update_mini_batch(netw, size(tr_x)[2], mini_batch_x, mini_batch_y, eta, cf, lambda)

        println("epoch: ", i, " -- Accuracy on evaluation data: ", evaluateRes(netw, ev_x, ev_y))     #, " ", sum.(netw.weights - oldweights)
        # println(total_cost(netw, tr_x, tr_y, cf, lambda))
        acc_tr = vcat(acc_tr, evaluateRes(netw, tr_x, tr_y))
        acc_ev = vcat(acc_ev, evaluateRes(netw, ev_x, ev_y))
        acc_te = vcat(acc_te, evaluateRes(netw, te_x, te_y))
        append!(cost_te, total_cost(netw, te_x, te_y, cf, lambda))
        append!(cost_ev, total_cost(netw, ev_x, ev_y, cf, lambda))
        append!(cost_tr, total_cost(netw, tr_x, tr_y, cf, lambda))
    end
    return acc_tr, acc_ev, acc_te, cost_tr, cost_ev, cost_te
end

function backprop(a,y, netw, cf)
    nabCb = [zeros(size(i)) for i in netw.biases]
    nabCw = [zeros(size(i)) for i in netw.weights]
    activation = a
    activations = [a]
    zs = []
    for (b,w) in zip(netw.biases, netw.weights)
        z = w * activation .+ b
        append!(zs, z)
        activation = sigmoid.(z)
        append!(activations, [activation])
    end
    #delta = - (y - activations[size(activations)[1]]) .* sigmoid_prime.(zs[size(zs)[1]])
    # delta  = mseD(activations[size(activations)[1]], y, zs[size(zs)[1]])
    #delta = - (y - activations[size(activations)[1]])
    if cf == "MSE"
        delta = mseD(activations[size(activations)[1]], y, zs[size(zs)[1]])
    elseif cf == "Cross-Entropy"
        delta = crossEntD(activations[size(activations)[1]], y)
    end

    nabCb[size(nabCb)[1]] .= [sum(delta[i,:]) for i=1:size(delta)[1]]
    nabCw[size(nabCw)[1]] = delta * transpose(activations[size(activations)[1]-1])
    for l=1:(netw.numOfLayers-2)
        z = zs[size(zs)[1]-l]
        sp = sigmoid_prime.(z)
        delta = (transpose(netw.weights[size(netw.weights)[1]-l+1]) * delta) .* sp
        nabCb[size(nabCb)[1]-l] .= [sum(delta[i,:]) for i=1:size(delta)[1]]
        nabCw[size(nabCw)[1]-l] = delta * transpose(activations[size(activations)[1]-l-1])
    end
    return nabCb, nabCw
end

function update_mini_batch(netw, trainingSetSize, mini_batch_x, mini_batch_y, eta, cf, lambda)
    nabCb = [zeros(size(i)) for i in netw.biases]
    nabCw = [zeros(size(i)) for i in netw.weights]
    for (x,y) in zip(mini_batch_x,mini_batch_y)
        nabCbi, nabCwi = backprop(x, y, netw, cf)
        nabCb = [nb + dnb for (nb,dnb) in zip(nabCb, nabCbi) ]
        nabCw = [nw + dnw for (nw,dnw) in zip(nabCw, nabCwi) ]
    end
    oldweights = netw.weights
    # netw.weights = [w*(1-eta*lambda/sum(length.(netw.weights))) - eta/minibatchsize * nabC for (w,nabC) in zip(netw.weights, nabCw)] #add regularization w*(1-eta*lambda/N) sum(length.(net.weights))
    # netw.weights = [w*(1-eta*lambda) - eta/minibatchsize * nabC for (w,nabC) in zip(netw.weights, nabCw)]
    netw.weights = [w*(1-eta*lambda/trainingSetSize) - eta/size(mini_batch_x)[1] * nabC for (w,nabC) in zip(netw.weights, nabCw)] #1/num_of_training_batches
    net.biases = [b - eta/size(mini_batch_x)[1] * nabC for (b,nabC) in zip(net.biases, nabCb)]
    return oldweights
end

function feedforward(netw, a)
    for (b, w) in zip(netw.biases, netw.weights)
        a = sigmoid.((w*a)+b)
    end
    return a
end

function evaluateRes(netw, test_data_x, test_data_y)
    as = [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]
    test_results = [as[argmax(feedforward(netw,test_data_x[:,i]))]==as[argmax(test_data_y[:,i])] for i=1:size(test_data_x)[2]]    #size(test_data_x)[2]
    return sum(test_results)/size(test_data_x)[2]
end

function sigmoid(z)
    f = (1 + exp(-z))^(-1)
end

function sigmoid_prime(z)
    return sigmoid(z)*(1-sigmoid(z))
end

function meansqerr(a, y)
    r = (norm(y .- a))^2 * 0.5
    # r = (norm(y .- vecRes.(argmax(a)[1])))^2 * 0.5
    return r
end

function mseD(a, y, z)
    return -(y .- a) .* sigmoid_prime.(z)
end

function crossEnt(a,y)
    r1 = y.*log.(a) + (1 .- y).* log.(1 .- a )
    r2 = -sum(r1[.!isnan.(r1)])
    ce = r2 < 10^7 ? r2 : 10^7
    # ce = -sum(y.*log.(vecRes.(argmax(a)[1])) + (1 .- y).* log.(1 .- vecRes.(argmax(a)[1]) ))
    return ce
end

function crossEntD(a, y)
    return a .- y
end

function total_cost(netw, tr_x, tr_y, cf, lambda)
    if cf == "MSE"
        test_result = sum( meansqerr(feedforward(netw,tr_x[:,i]), tr_y[:,i]) for i=1:size(tr_x)[2])/size(tr_x)[2] + lambda*(sum(netw.weights[1] .^2) + sum(netw.weights[2] .^2))/(2*size(tr_x)[2])
    elseif cf == "Cross-Entropy"
        test_result = sum( crossEnt(feedforward(netw,tr_x[:,i]), tr_y[:,i]) for i=1:size(tr_x)[2])/size(tr_x)[2] + lambda*(sum(netw.weights[1] .^2) + sum(netw.weights[2] .^2))/(2*size(tr_x)[2])
    end
    # test_result = sum( meansqerr(feedforward(netw,tr_x[:,i]), tr_y[:,i]) for i=1:size(tr_x)[2])/size(tr_x)[2]
    #test_result = size(feedforward(netw,tr_x[:,1]))
    #test_result = sum((feedforward(netw,tr_x[:,i]) - tr_y[:,i])' * (feedforward(netw,tr_x[:,i]) - tr_y[:,i])  for i=1:size(tr_x)[2])
    #test_results = [as[argmax(feedforward(netw,tr_x[:,i]))] - tr_y[i] for i=1:size(tr_x)[2]]    #size(test_data_x)[2]
    #* (feedforward(netw,tr_x[:,i]) - tr_y[i])' - tr_y[i]
    return test_result
end

#########
# MBsize = 15
# L rate = 0.001
# NN 784 15 10
########

net = networkF([784,30,10], "Default Weight")
tr_x, tr_y, te_x, te_y, ev_x, ev_y = loadData(size(net.weights[2])[1], 1000, 100, 100)
acc_tr, acc_ev, acc_te, cost_tr, cost_ev, cost_te = SGD(net, 30, 10, 1, "Cross-Entropy", 20, tr_x, tr_y, te_x, te_y, ev_x, ev_y)
acc_tr, acc_ev, acc_te, cost_tr, cost_ev, cost_te = SGD(net, 300, 15, .001, "MSE", 0.1)

acc_trO, acc_ev0, acc_teO, cost_trO, cost_ev0, cost_teO = acc_tr, acc_ev, acc_te, cost_tr, cost_ev, cost_te
########READFiles
function loadData(vecsize, trainSize, evSize, testSize)
    train_x, train_y = MNIST.traindata()
    test_x,  test_y  = MNIST.testdata()

    tr_x = reshape(train_x,(784,60000))
    test = vecRes.(train_y, vecsize)
    tr_y = test[1]
    for i=2:size(test)[1]
        tr_y = hcat(tr_y,test[i])
    end

    ev_x, ev_y = tr_x[:,1:evSize], tr_y[:,1:evSize]
    tr_x, tr_y = tr_x[:,evSize+1:evSize+trainSize], tr_y[:,evSize+1:evSize+trainSize]

    te_x = reshape(test_x,(784,10000))  #[reshape(test_x[:,:,1],(784,1)) for i=1:size(test_x)[3]]
    test = vecRes.(test_y, vecsize)
    te_y = test[1]
    for i=2:size(test)[1]
        te_y = hcat(te_y,test[i])
    end

    te_x, te_y = te_x[:,1:testSize], te_y[:,1:testSize]

    # tr_x = rand(4,10)
    # tr_y = rand(2,10)
    # te_x = rand(4,2)
    # te_y = rand(2,2)
    return tr_x, tr_y, te_x, te_y, ev_x, ev_y
end

function vecRes(args, vecsize)
    e = zeros(vecsize) #zeros(10,1)
    e[args+1]=1.0
    return e
end

############################################

colorview(Gray, hcat(train_x[:,:,1], train_x[:,:,24]))
tr_x, tr_y, te_x, te_y, ev_x, ev_y = loadData(size(net.weights[2])[1], 1000, 100, 100)

####################################################

plot([scatter(x=1:size(acc_te)[1], y=[acc_te[i][1] for i=1:size(acc_te)[1]], name="accuracy on test set"), scatter(x=1:size(acc_tr)[1], y=[acc_tr[i][1] for i=1:size(acc_tr)[1]], name="accuracy on training set"), scatter(x=1:size(acc_ev)[1], y=[acc_ev[i][1] for i=1:size(acc_ev)[1]], name="accuracy on ev set")], Layout(; yaxis_range=[0.0,.92], xaxis=attr(title="epochs"), yaxis=attr(title="accuracy")))
plot([scatter(x=1:size(acc_te)[1], y=[acc_te[i][2][1] for i=1:size(acc_te)[1]], name="w"), scatter(x=1:size(acc_te)[1], y=[acc_te[i][2][2] for i=1:size(acc_te)[1]], name="b")], Layout(; xaxis=attr(title="epochs"), yaxis=attr(title="parameters")))
# plot([i for i=1:size(d1)[1]], [d1[i][2][2] for i=1:size(d1)[1]])

plot([scatter(x=1:size(cost_tr)[1], y=cost_tr, mode="lines+marker", name="train"), scatter(x=1:size(cost_te)[1], y=cost_te, mode="lines+markers", name="test set"), scatter(x=1:size(cost_ev)[1], y=cost_ev, mode="lines+markers", name="ev set")], Layout(; xaxis=attr(title="epochs"), yaxis=attr(title="cost function")))

, xaxis_range=[0.75, 5.25], yaxis_range=[0, 8] title="cost function",

writedlm("./DL/cost_tr-ev-te.dat", [cost_tr cost_ev cost_te])


tr_x, tr_y, te_x, te_y, ev_x, ev_y = loadData(size(net.weights[2])[1])

net = networkF([784,30,10], "Large Weight")
