import numpy as np

def some_fun(x):
    if x == []: 
        return 0
    else:
        return 1 + some_fun(x[1:] )

def quick_sort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        smaller, bigger = [ ] , [ ]
        for element in array[1:]:
            
            if element <= pivot:
                smaller.append(element) 
            else: 
                bigger.append(element)
        
        return quick_sort(smaller) + [pivot] + quick_sort(bigger)
        
if __name__ == "__main__":
    
    x = list([12, 1, 2, 3, 40, 1, 22, 40])
    
    print(quick_sort(x)) 