class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self,x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2