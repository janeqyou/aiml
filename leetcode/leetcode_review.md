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
```
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

### Recursion II
#### Merge Sort
- Time Complexity
    +  Recursively divide the input list into two sublists, until a sublist with single element remains. This dividing step computes the midpoint of each of the sublists, which takes {O(1)} time. This step is repeated N times until a single element remains, therefore the total time complexity is O(LogN).
    +  In the merging process, each level still has N elements. When merging two segments, the process needs to go through each elements to arrive at the final list. So each level takes O(N) to merge. There are total O(LogN) levels. 

- Space Complexity
    + This is not an inplace sort but calls recursion. So the space complexity is O(N) 

#### Divide and Conquer 
- Divide and Conquer. Naturally implemented as recurrsion. Another subtle difference that tells a divide-and-conquer algorithm apart from other recursive algorithms is that we break the problem down into *two or more* subproblems in the divide-and-conquer algorithm
![divide-conquer](/Users/qxy001/Documents/personal_src/aiml/leetcode/divide-n-conquer.png)

#### Backtracking
- During recurrsion,once we can determine if a certain node cannot possibly lead to a valid solution, we abandon the current node and backtrack to its parent node to explore other possibilities. Backtracking reduced the number of steps taken to reach the final result. This is known as pruning the recursion tree because we don't take unnecessary paths.
- Template: using N queen problem as an example. The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. A queen can attack any piece that is situated at the same row, column or diagonal of the queue.

    #### N Queen structure 
```def backtrack_nqueen(row = 0, count = 0):
    
    for col in range(n):       
        # iterate through columns at the curent row.
        if is_not_under_attack(row, col):
            # explore this partial candidate solution, and mark the attacking zone
            place_queen(row, col)
            if row + 1 == n:
                # we reach the bottom, i.e. we find a solution!
                count += 1
            else:
                # we move on to the next row, assuming current partial solution
                # is correct 
                count = backtrack(row + 1, count)
            # backtrack, i.e. remove the queen and remove the attacking zone.
            # meaning we have reached or a solution, remove all the constraints
            # happened after this step
            remove_queen(row, col)
    return count
```
    #### general template 
```def backtrack(candidate):

        if find_solution(candidate):
            output(candidate)
            return
        # iterate all possible candidates.
        for next_candidate in list_of_candidates:
            if is_valid(next_candidate):
                # try this partial candidate solution
                place(next_candidate)
                # given the candidate, explore further.
                backtrack(next_candidate)
                # backtrack
                remove(next_candidate)
```
