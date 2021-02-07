### Recursion I 
#### Time complexity
- Its time complexity ${\mathcal{O}(T)}$ which is typically the product of the number of recursion invocations (denoted as $R$) and the time complexity of calculation (denoted as ${\mathcal{O}(s)}$) that incurs along with each recursion call: $\mathcal{O}(T) = R * \mathcal{O}(s)$
-  If there are more than one component that requires recursive calls, resort to recursive tree. It is a tree used to denote the execution flow of a recursive function in particular. Each node in the tree represents an invocation of the recursive function. Therefore, the total number of nodes in the tree corresponds to the number of recursion calls during the execution. An N depth binary three has $2^n$ nodes. So in this case the time complexity is calculating every node $\mathcal{O}(2^n)$

#### Space complexity
- The recursion related space refers to the memory cost that is incurred directly by the recursion, i.e. the stack to keep track of recursive function calls. 
    + The returning address of the function call. Once the function call is completed, the program must know where to return to, i.e. the line of code after the function call.
    + The parameters that are passed to the function call. 
    + The local variables within the function call.
If the recursive calls will chain up to n times, where n is the size of the input string. So the space complexity of this recursive algorithm is ${\mathcal{O}(n)}$.
- Non recursion related space include global variable (in heap) and space for memoization

#### Tail recursion
- non tail recursion: there is additional computation after recursive call returns. Need $\mathcal{O}(n)$ memory allocation for recursion
```python

def sum_non_tail_recursion(ls):

    if len(ls) == 0:
        return 0
    
    -- not a tail recursion because it does some computation after the recursive call returned.
    return ls[0] + sum_non_tail_recursion(ls[1:])
```

- tail recursion: recursive call directly returns to the upper level 

``` 
    def sum_tail_recursion(ls):
    
        def helper(ls, acc):
            if len(ls) == 0:
                return acc
            # this is a tail recursion because the final instruction is a recursive call.
            return helper(ls[1:], ls[0] + acc)  
    return helper(ls, 0)
```
The benefit of having tail recursion is that it could avoid the accumulation of stack overheads during the recursive calls, since the system could *reuse a fixed amount space* in the stack for each recursive call.

#### Recursion II 
