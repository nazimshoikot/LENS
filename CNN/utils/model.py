from utils.convolution_functions import conv_forward, conv_backward, conv_SDLM
from utils.max_pooling_functions import pool_forward, pool_backward
from utils.activation_functions import *
from utils.FC_weights import get_initial_weight

# import initialize, update
import numpy as np

def initialize(kernel_shape):
    b_shape = (1, 1, 1, kernel_shape[-1]) if len(kernel_shape) == 4 else (kernel_shape[-1],)
    mu, sigma = 0, 0.1
    weight = np.random.normal(mu, sigma, kernel_shape)
    bias = np.ones(b_shape) * 0.01
    return weight, bias

# update for the weights 
# regularization can be changed if we want it to converge faster
# overfit_constant can be set if we want to consider overfitting
def update(weight, bias, dW, db, vw, vb, lr, regularization=0.9, overfit_constant=0.6):
    # print("\n\nUPDATING WEIGHTS")
    # print("Inputs:\nweights:{}\nbias:{}\ndW:{}\ndb:{}\nvw:{}\nvb:{}".format(weight,bias,dW,db,vw,vb))
    vw_u = regularization*vw - overfit_constant*lr*weight - lr*dW
    vb_u = regularization*vb - overfit_constant*lr*bias   - lr*db
    
    weight_u = weight + vw_u
    bias_u   = bias   + vb_u

    # print("Returning:\nweight:{}\nbias:{}\nvw:{}\nvb:{}\n\n".format(weight_u, bias_u, vw_u,vb_u))
    return weight_u, bias_u, vw_u, vb_u 

class ConvLayer(object):
    def __init__(self, kernel_shape, hparameters):
        self.hparameters = hparameters
        self.weight, self.bias = initialize(kernel_shape)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        
    def forward_prop(self, input_map):
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.hparameters)
        return output_map
    
    def backward_prop(self, dZ, lr_global):
        # dZ = input from prev layer
        # lr_global is the global learning rate
        dA_prev, dW, db = conv_backward(dZ, self.cache)

        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr)

        return dA_prev

    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z, mu, lr_global):
        d2A_prev, d2W = conv_SDLM(d2Z, self.cache)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev 

class MaxPoolingLayer(object):
    def __init__(self, hparameters):
        self.hparameters = hparameters
        
    def forward_prop(self, input_map):   # n,28,28,6 / n,10,10,16
        A, self.cache = pool_forward(input_map, self.hparameters)
        return A
    
    def backward_prop(self, dA):
        dA_prev = pool_backward(dA, self.cache)
        return dA_prev
    
    def SDLM(self, d2A):
        d2A_prev = pool_backward(d2A, self.cache)
        return d2A_prev

class Activation(object):
    def __init__(self):    
        # select the ReLU function as our activation function
        self.act   = ReLU
        self.d_act = d_ReLU
        
    def forward_prop(self, input_image): 
        self.input_image = input_image
        return self.act(input_image)
    
    def backward_prop(self, dZ):
        dA = np.multiply(dZ, self.d_act(self.input_image)) 
        return dA
   
class FCLayer(object):
    def __init__(self, weight_shape): 
        
        # Initialization
        self.v_w, self.v_b = np.zeros(weight_shape), np.zeros((weight_shape[-1],))
        self.weight, self.bias = initialize(weight_shape)
        
    def forward_prop(self, input_array):
        self.input_array = input_array  
        return np.matmul(self.input_array, self.weight) 
        
    def backward_prop(self, dZ, lr_global):
        dA = np.matmul(dZ, self.weight.T)               
        dW = np.matmul(self.input_array.T, dZ)         
        db = np.sum(dZ.T, axis=1)                       
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr)
        return dA

    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z, mu, lr_global):
        d2A = np.matmul(d2Z, np.power(self.weight.T,2))
        d2W = np.matmul(np.power(self.input_array.T,2), d2Z)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A

initial_FC_weights = get_initial_weight()
class OutputLayer(object):
    def __init__(self, weight):        
        self.weight = weight  # (10, 84)
        
    def forward_prop(self, input_array, label, mode):
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label,:] 
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)
                        
            return np.sum(np.squeeze(loss))
        else:
            subtract_weight = (input_array[:,np.newaxis,:] - np.array([self.weight]*input_array.shape[0]))

            rbf_class = np.sum(np.power(subtract_weight,2), axis=2)
            class_pred = np.argmin(rbf_class, axis=1)
            error01 = np.sum(label != class_pred)

            return error01, class_pred
        
    def backward_prop(self):
        dy_predict = -self.weight_label + self.input_array  
        return dy_predict
    
    def SDLM(self):
        return np.ones(self.input_array.shape)

# LeNet5 object
class LeNet5(object):
    def __init__(self):
        # YOUR IMPLEMETATION
        # define the shapes
        kernel_shape = {"C1": (5,5,1,6),
                        "C3": (5,5,6,16),    
                        "C5": (5,5,16,120),  
                        "F6": (120,84),
                        "F7": (84,10)}
        
        # define the different parameters
        hparameters_convlayer = {"stride": 1, "pad": 0}
        hparameters_pooling   = {"stride": 2, "f": 2} 
        self.C1 = ConvLayer(kernel_shape["C1"], hparameters_convlayer)
        self.S2 = MaxPoolingLayer(hparameters_pooling)
        self.a1 = Activation()
        self.C3 = ConvLayer(kernel_shape["C3"], hparameters_convlayer)
        self.S4 = MaxPoolingLayer(hparameters_pooling)
        self.a2 = Activation()
        self.C5 = ConvLayer(kernel_shape["C5"], hparameters_convlayer)
        self.a3 = Activation()
        self.F6 = FCLayer(kernel_shape["F6"])
        self.a4 = Activation()
        self.Output = OutputLayer(initial_FC_weights)

    def Forward_Propagation(self, input_image, input_label, mode):
        # YOUR IMPLEMETATION
        self.label = input_label
        self.C1_FP = self.C1.forward_prop(input_image)
        self.S2_FP = self.S2.forward_prop(self.C1_FP)
        self.a1_FP = self.a1.forward_prop(self.S2_FP)
        self.C3_FP = self.C3.forward_prop(self.a1_FP)
        self.S4_FP = self.S4.forward_prop(self.C3_FP)
        self.a2_FP = self.a2.forward_prop(self.S4_FP)
        self.C5_FP = self.C5.forward_prop(self.a2_FP)
        self.a3_FP = self.a3.forward_prop(self.C5_FP)
        self.flatten = self.a3_FP[:,0,0,:]
        self.F6_FP = self.F6.forward_prop(self.flatten)
        self.a4_FP = self.a4.forward_prop(self.F6_FP)
        # when mode = 'train', output sum of loss over minibatch
        # when mode = test, output the predicted class
        out  = self.Output.forward_prop(self.a4_FP, input_label, mode) 
        return out 
        
    def Back_Propagation(self, lr_global):
        self.SDLM(0.02, lr_global)
        # YOUR IMPLEMETATION
        dy_pred = self.Output.backward_prop()
        a4_BP = self.a4.backward_prop(dy_pred)
        F6_BP = self.F6.backward_prop(a4_BP, lr_global)
        reverse_flat = F6_BP[:,np.newaxis,np.newaxis,:]
        a3_BP = self.a3.backward_prop(reverse_flat)
        C5_BP = self.C5.backward_prop(a3_BP, lr_global)
        a2_BP = self.a2.backward_prop(C5_BP)
        S4_BP = self.S4.backward_prop(a2_BP)
        C3_BP = self.C3.backward_prop(S4_BP, lr_global)
        a1_BP = self.a1.backward_prop(C3_BP)
        S2_BP = self.S2.backward_prop(a1_BP)
        C1_BP = self.C1.backward_prop(S2_BP, lr_global)
    
    # Stochastic Diagonal Levenberg-Marquaedt method for determining the learning rate 
    def SDLM(self, mu, lr_global):
        d2y_pred = self.Output.SDLM()
        F6_SDLM = self.F6.SDLM(d2y_pred, mu, lr_global)
        reverse_flatten = F6_SDLM[:,np.newaxis,np.newaxis,:]
        C5_SDLM = self.C5.SDLM(reverse_flatten, mu, lr_global)
        S4_SDLM = self.S4.SDLM(C5_SDLM)
        C3_SDLM = self.C3.SDLM(S4_SDLM, mu, lr_global)
        S2_SDLM = self.S2.SDLM(C3_SDLM)
        C1_SDLM = self.C1.SDLM(S2_SDLM, mu, lr_global)



# =================LeNet5CE model and its classes and functions=======================


def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    # print("Predicted: ", Y_pred[:5])
    # print("Actual: ", Y_true[:5])
    # exit()
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N
      
class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        #print("Softmax: _forward")
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        # print("Predicted: ", Y_pred[:5])
        # print("Actual: ", Y_true[:5])
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        loss = NLLLoss(Y_pred, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout

class CeReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def forward_prop(self, X):
        #print("ReLU: _forward")
        out = np.maximum(0, X)
        self.cache = X
        return out

    def backward_prop(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

class Conv():
    """
    Conv layer
    """
    def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
        self.Cin = Cin
        self.Cout = Cout
        self.F = F
        self.S = stride
        #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.cache = None
        self.pad = padding

    def _forward(self, X):
        X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
        (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1
        W_ = W - self.F + 1
        Y = np.zeros((N, self.Cout, H_, W_))

        for n in range(N):
            for c in range(self.Cout):
                for h in range(H_):
                    for w in range(W_):
                        Y[n, c, h, w] = np.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]

        self.cache = X
        return Y

    def backward_prop(self, dout,lr):
        # dout (N,Cout,H_,W_)
        # W (Cout, Cin, F, F)
        X = self.cache
        (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1
        W_ = W - self.F + 1
        W_rot = np.rot90(np.rot90(self.W['val']))

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['val'].shape)
        db = np.zeros(self.b['val'].shape)

        # dW
        for co in range(self.Cout):
            for ci in range(Cin):
                for h in range(self.F):
                    for w in range(self.F):
                        dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

        # db
        for co in range(self.Cout):
            db[co] = np.sum(dout[:,co,:,:])

        dout_pad = np.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
        #print("dout_pad.shape: " + str(dout_pad.shape))
        # dX
        for n in range(N):
            for ci in range(Cin):
                for h in range(H):
                    for w in range(W):
                        #print("self.F.shape: %s", self.F)
                        #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                        dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

        return dX

class MaxPool():
    def __init__(self, F, stride):
        self.F = F
        self.S = stride
        self.cache = None

    def _forward(self, X):
        # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
        (N,Cin,H,W) = X.shape
        F = self.F
        W_ = int(float(W)/F)
        H_ = int(float(H)/F)
        Y = np.zeros((N,Cin,W_,H_))
        M = np.zeros(X.shape) # mask
        for n in range(N):
            for cin in range(Cin):
                for w_ in range(W_):
                    for h_ in range(H_):
                        Y[n,cin,w_,h_] = np.max(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)])
                        i,j = np.unravel_index(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(), (F,F))
                        M[n,cin,F*w_+i,F*h_+j] = 1
        self.cache = M
        return Y

    def backward_prop(self, dout):
        M = self.cache
        (N,Cin,H,W) = M.shape
        dout = np.array(dout)
        #print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
        dX = np.zeros(M.shape)
        for n in range(N):
            for c in range(Cin):
                #print("(n,c): (%s,%s)" % (n,c))
                dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
        return dX*M

class FC():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        #print("Build FC")
        self.cache = None
        #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        #print("FC: _forward")
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def backward_prop(self, dout,lr):
        #print("FC: _backward")
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        self._update_params(lr)
        return dX

    def _update_params(self, lr):
        # Update the parameters
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']

class LeNet5CE(object):
    def __init__(self):
        self.C1 = Conv(1,6,5)
        self.a1 = CeReLU()
        self.S2 = MaxPool(2,2)
        
        self.C3 = Conv(6,16,5) 
        self.a2 = CeReLU()
        self.S4 = MaxPool(2,2)
        
        self.C5 = Conv(16,120,5)
        self.a3 = CeReLU()

        self.F6 = FC(120,84)
        self.a4 = CeReLU()

        self.F7 = FC(84,10)
        self.Softmax = Softmax()

        self.loss_function = CrossEntropyLoss()
        self.prev_shape = None
        
    def Forward_Propagation(self, input_image, input_label, mode): 
        self.label = input_label
        oneHotLabel = MakeOneHot(input_label, 10)   
        input_image = np.swapaxes(input_image, 1,3)
        self.C1_FP = self.C1._forward(input_image)
        self.S2_FP = self.S2._forward(self.C1_FP)
        self.a1_FP = self.a1.forward_prop(self.S2_FP)
        
        self.C3_FP = self.C3._forward(self.a1_FP)

        self.S4_FP = self.S4._forward(self.C3_FP)
        self.a2_FP = self.a2.forward_prop(self.S4_FP)
        
        self.C5_FP = self.C5._forward(self.S4_FP)
        self.a3_FP = self.a3.forward_prop(self.C5_FP)
        self.prev_shape = self.a3_FP.shape
        self.flatten = self.a3_FP.reshape(input_image.shape[0],-1)
        self.F6_FP = self.F6._forward(self.flatten)
        self.a4_FP = self.a4.forward_prop(self.F6_FP)  
        self.F7_FP = self.F7._forward(self.a4_FP)
        self.soft = self.Softmax._forward(self.F7_FP)

        if (mode == "train"):
            loss, dout = self.loss_function.get(self.soft, oneHotLabel)
            return loss, dout

        else:
            Y_pred = self.soft
            Y_train = input_label
            predictions = np.argmax(Y_pred, axis=1)
            result = predictions - Y_train
            result = list(result)

            out = (len(result) - result.count(0))
            return out, predictions
        
    def Back_Propagation(self, dout, lr_global):
        # YOUR IMPLEMETATION
        F7_BP = self.F7.backward_prop(dout, lr_global)
        a4_BP = self.a4.backward_prop(F7_BP)
        F6_BP = self.F6.backward_prop(a4_BP, lr_global)
        reverse_flat = F6_BP.reshape(self.prev_shape)
        a3_BP = self.a3.backward_prop(reverse_flat)
        C5_BP = self.C5.backward_prop(a3_BP, lr_global)
        a2_BP = self.a2.backward_prop(C5_BP)
        S4_BP = self.S4.backward_prop(a2_BP)
        C3_BP = self.C3.backward_prop(S4_BP, lr_global)
        a1_BP = self.a1.backward_prop(C3_BP)
        S2_BP = self.S2.backward_prop(a1_BP)
        C1_BP = self.C1.backward_prop(S2_BP, lr_global)
    

