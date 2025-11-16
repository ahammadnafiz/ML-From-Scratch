import math
from typing import Any, Dict, List, Optional, Tuple, Union


class Variable:
    def __init__(self, data: float, _childred: tuple = (), _op: str = '', name: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self.name = name
        
        self._backward = lambda : None # Function to compute gradients
        self._prev = set(_childred) # Previous nodes in computional graph
        self._op = _op # Operation that created this node
        
    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else ''
        return f"Variable(data={self.data:.6f}, grad={self.grad:.6f}{', name=' + name_str if name_str else ''})"
        
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')
        
        def _backward():
            # d(a + b)/da = 1, d(a + b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')
        
        def _backward():
            # d(a * b)/da = b, d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Variable(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def _neg__(self):
        return self * -1
    
    
    def __repr__(self):
        return f"Variable(data={self.data:.6f}, grad={self.grad:.6f}, name='{self.name}')"
    
    def __str__(self):
        return f"Variable(data={self.data:.6f}, grad={self.grad:.6f}, name='{self.name}')"
    
    def __len__(self):
        return len(self.data)
            
