x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1 # random guess

#Our model for forword pass
def forward(x):
    return x*w

#Loss funcion
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


#Compute the Gradient
def gradient(x, y):
    return 2 * x * (x * w - y)

#Beore traning
print("Before training X =", 4 , "Y = ",forward(4))


# Training loop
for epoch in range(1000):      
       for x_val, y_val in zip(x_data, y_data):
           grad = gradient(x_val, y_val)
           w = w - 0.01 * grad
           print("/t", x_val, y_val, grad)
           l = (loss(x_val, y_val))
       print ("Progress:", epoch, "w=", w, "Loss", l)

#After traning
print("After training X =", 4 , "Y = ",forward(4))