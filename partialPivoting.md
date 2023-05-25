---
output:
  html_document:
    css: projectCSS.css
    toc: true
    toc_float: true
editor_options: 
  markdown: 
    wrap: 72
---

## **M 340L Coding Project**

#### Jasmine Xu (jcx67)

### **Computational Section**

> a.  Code up row reduction with partial pivoting (choosing the pivot to
>     be the largest in absolute magnitude in the pivot search) for an
>     $m×n$ matrix.

#### Code

``` python
import numpy as np
from numpy import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, linewidth=200)

# Aim: to code up row reduction with partial pivoting (choosing the pivot
# to be the largest in absolute magnitude in the pivot search) for an
# m × n matrix

## an example 4 x 4 matrix, all entries between 0 and 25
randomIntMat = np.array(random.randint(25, size = (4, 4)),
                        dtype=np.float64)
## in the case of part b, we would just pass randomMatrix into the echelon function
randomMatrix = random.rand(100, 10)

# Function that converts an augmented matrix into echelon form by identifying
# pivots and creating zeros below them via row-replacement operations
def echelon(mat):
    print(f"Starting matrix: \n{mat}\n")
    # m rows, n cols
    m, n = mat.shape
    pcount = 0
    # for each row in matrix...
    for i in range(m - 1):
        print(f"---- Pivot Row {i} ----\n")
        # if no pivot is available in pivot col, move to the next col
        if all(mat[i:, pcount] == 0):
            pcount += 1
        # if the element in the pivot position in row i is 0, switch row i w/ the
        # first row after that contains a nonzero element in the col
        if mat[i][pcount] == 0:
            nzRow = np.nonzero(mat[i:, pcount])[0][0]
            print(f"Pivot is zero: swapping rows {i} and {nzRow + i}:")
            mat[[i, nzRow + i]] = mat[[nzRow + i, i]]
            print(mat, "\n")
        #### PART WITH PARTIAL PIVOTING ####
        # if the pivot is NOT the maximum value (in magnitude) in its column, switch
        # pivot row with the row containing the max
        if abs(mat[i][pcount]) != max(abs(mat[i:, pcount])):
            absCol = abs(mat[i:, pcount])
            # the (first) row number that contains the largest value
            rowWithLargest = np.where(absCol == max(absCol))[0][0]
            # swapping the current row with the row that contains max value
            print(f"Partial Pivoting: swapping rows {i} and {rowWithLargest + i}")
            mat[[i, rowWithLargest + i]] = mat[[rowWithLargest + i, i]]
            print(mat, "\n")
        # for every row following the pivot row, row-replacement operations are performed
        for j in range(i + 1, m):
            # only change the row if the value isn't already zero
            if mat[j][pcount] != 0:
                mult = mat[j][pcount] / mat[i][pcount] * -1
                mat[j][pcount:] += mat[i][pcount:] * mult
        pcount += 1
        print("Row-replaced matrix:")
        print(mat, "\n")
        # break when the number of pivots is equal to the number of cols
        if pcount == n:
            break
    return mat


finalMatrix = echelon(randomIntMat)
print("FINAL matrix in echelon form, WITH partial pivoting:\n", finalMatrix)
```

#### Sample Output for a random 4 x 4 matrix

    Starting matrix:
    [[ 2. 10.  8.  8.]
     [ 8. 11. 18.  1.]
     [ 3.  6. 21. 22.]
     [ 0. 24.  2. 23.]]

    ---- Pivot Row 0 ----

    Partial Pivoting: swapping rows 0 and 1
    [[ 8. 11. 18.  1.]
     [ 2. 10.  8.  8.]
     [ 3.  6. 21. 22.]
     [ 0. 24.  2. 23.]]

    Row-replaced matrix:
    [[ 8.    11.    18.     1.   ]
     [ 0.     7.25   3.5    7.75 ]
     [ 0.     1.875 14.25  21.625]
     [ 0.    24.     2.    23.   ]]

    ---- Pivot Row 1 ----

    Partial Pivoting: swapping rows 1 and 3
    [[ 8.    11.    18.     1.   ]
     [ 0.    24.     2.    23.   ]
     [ 0.     1.875 14.25  21.625]
     [ 0.     7.25   3.5    7.75 ]]

    Row-replaced matrix:
    [[ 8.         11.         18.          1.        ]
     [ 0.         24.          2.         23.        ]
     [ 0.          0.         14.09375    19.828125  ]
     [ 0.          0.          2.89583333  0.80208333]]

    ---- Pivot Row 2 ----

    Row-replaced matrix:
    [[ 8.         11.         18.          1.        ]
     [ 0.         24.          2.         23.        ]
     [ 0.          0.         14.09375    19.828125  ]
     [ 0.          0.          0.         -3.27198817]]

    FINAL matrix in echelon form, WITH partial pivoting:
     [[ 8.         11.         18.          1.        ]
     [ 0.         24.          2.         23.        ]
     [ 0.          0.         14.09375    19.828125  ]
     [ 0.          0.          0.         -3.27198817]]

> b.  Test with an $100×10$ matrix with random (uniformly distributed
>     between 0 and 1) entries. Print the result of row reduction to
>     echelon form (not reduced echelon form). Thus, your output should
>     be an upper triangular matrix.

Here is the final output when we test with a 100 x 10 matrix with random
floating point entries between 0 and 1:

    FINAL matrix in echelon form, WITH partial pivoting:
     [[ 0.97956735  0.80866861  0.80383857  0.23629158  0.11091572  0.04337259  0.69160915  0.28188561  0.56335725  0.44093573]
     [ 0.          0.84871779  0.66259961  0.72254739  0.75968144  0.43360931  0.20258131 -0.00428368  0.24816562  0.24638376]
     [ 0.          0.          0.86317753  0.75681823  0.08783394  0.58986982  0.42532613 -0.02328761  0.29293868  0.8975888 ]
     [ 0.          0.          0.          1.23300832  0.22566225  0.58133716 -0.03203446  0.05966169  0.31193786  0.7642666 ]
     [ 0.          0.          0.          0.          1.26969011  0.52622968 -0.52463392  0.53735257 -0.17669847  0.04634519]
     [ 0.          0.          0.          0.          0.          0.94141687  0.60048248  0.97862588  0.41673192  0.29463478]
     [ 0.          0.          0.          0.          0.          0.          1.19971518  0.53905506  0.40435983  0.28021716]
     [ 0.          0.          0.          0.          0.          0.          0.         -1.04797183 -0.39664274 -0.03739307]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.71323635 -0.1467647 ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.         -1.0919607 ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.         -0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.         -0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.         -0.          0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.         -0.         -0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.         -0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.         -0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.         -0.        ]
     [ 0.         -0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.         -0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.         -0.        ]
     [ 0.          0.          0.          0.         -0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.         -0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.         -0.        ]
     [ 0.          0.          0.          0.          0.          0.         -0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.         -0.          0.         -0.          0.         -0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.         -0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.         -0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.         -0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.        ]]

> c.  Propose a way to speed up the algorithm you just wrote, which
>     still reduces the $m×n$ matrix to echelon form with partial
>     pivoting for a general matrix.

-   The 2nd and 3rd conditional blocks in the `echelon` function are
    probably a little redundant, in that if the 2nd block finds that the
    entry in the pivot position (`i`, `pcount`) is zero, it switches it
    with the next row containing a nonzero in position `pcount`, but
    this row might not be the row with the pivot of largest magnitude,
    so another row swap would have to be completed in the 3rd block.
    This should not be too difficult to fix, but it also shouldn't have
    a huge impact on the runtime in the first place.
-   More generally, row-replacement (Gaussian elimination) is inherently
    sequential, and so it makes sense to parallelise; but at the same
    time, this is a difficult task, considering how the entries in one
    row are dependent upon the rows above it.

------------------------------------------------------------------------

### **Theoretical Section**

> a.  Count the floating point operations FLOPs by hand required in
>     turning an $n × n$ matrix into echelon form. Hint: How to count
>     flops. Each multiplication, division, subtraction, and addition
>     counts as one floating point. Usually this number in the end is
>     reported using order notation. That is, if we have $O(n)$ (read:
>     "order n") it means, in a nutshell, that the process scales no
>     worse than some constant times $n$ as the number $n$ increases to
>     infinity.

(see file upload)

> b.  Show that Gaussian elimination is $O(n^3)$ by printing out in your
>     code how long it takes to do a $5×5$, $50×50$ and $500×500$ matrix
>     (using random entries uniformly distributed between 0 and 1 as
>     above).

I added a `testingRuntimes()` function to the code:

``` python
# testing runtimes for 5 x 5, 50 x 50, and 500 x 500
def testingRuntimes():
    # the array storing the runtimes for each matrix of size n
    runtimes = []
    for n in [5, 50, 500]:
        # matrix of size n with random entries between 0 and 1
        timedMatrix = random.rand(n, n)
        start = timer()
        echelon(timedMatrix)
        end = timer()
        runtime = end - start
        runtimes.append(runtime)
        print(f"Runtime for n = {n}: {runtime}\n")
    print(runtimes)
    print(f"5 -> 50: {runtimes[1] / runtimes[0]}\n"
          f"50 -> 500: {runtimes[2] / runtimes[1]}\n")
```

Here is the important part of the output:

    [0.012152800016338006, 0.17640870000468567, 2.6623876999947242]
    5 -> 50: 14.515889323244437
    50 -> 500: 15.092156452170485

The above output is saying that it takes \~0.012 seconds, \~0.176
seconds, and \~2.66 seconds to reduce a $5×5$, $50×50$ and $500×500$
matrix, respectively, to echelon form, and that runtime increases by a
factor of \~15 for each 10x increase in $n$. This definitely does not
conform to the expected runtimes, because we would have expected to see
a 1000x increase in runtime.

I ran this a few times and found that the factors by which runtime
increased varied pretty significantly (for example, it was \~15 in one
run and then \~110 the next) between each run. I was curious to see
whether these factors would converge to a certain value, so I ran
`testingRuntimes()` 100 times and plotted a histogram for the
distribution of the factor increase from n = 5 to n = 50, and also for n
= 50 to n = 500.

![](runtimeHistograms.png)

It looks like the most runtime increases are still by a factor of around
10-15 for both, which is pretty far from the expected growth rate. I
assume that this is mainly due to the sheer computational power of
modern computers. Also, the input matrices aren't particularly large,
and so hopefully we should see that the factor by which runtime
increases will converge towards what is expected as
$n\rightarrow \infty$.

(I also plotted histograms for when I ran `testingRuntimes()` 1000
times, but it was incredibly uninformative)

------------------------------------------------------------------------

### **Extension Problem**

#### With particularly large systems, often the system of equations is instead solved using, for example, the Jacobi method.

The Jacobi Method is an iterative scheme that aims to solve for
$\textbf x$ in the matrix equation $A\textbf x = \textbf b$ for some
given matrix $A$ and vector $\textbf b$ by allowing for some initial
guess $\textbf x_0$ to converge to the actual value of $\textbf x$.

> (a) Does this iterative method converge?

The Jacobi Method does *not* always converge; however, we can check for
convergence before running the method by confirming whether the spectral
radius of the iteration matrix is less than 1 (this is the
"if-and-only-if" condition).

This can be understood from the perspective of an error vector
$\textbf e_{i+1} = \textbf x_{i+1}-\textbf x$ - i.e., the deviation of
the iterated value of $\textbf x_{i+1}$ from the actual value of
$\textbf x$. If the Jacobi Method is to converge, then $\textbf e_{i+1}$
should approach zero as $i \rightarrow \infty$.

We are looking to solve $A\textbf x = \textbf b$ for $\textbf x$ through
an iterative method, where $\textbf x_i$ converges to $\textbf x$.

By the LU factorisation, $A\textbf x$ = $(D + L + U)\textbf x$ =
$\textbf b$.

We compute $\textbf e_{i+1}= \textbf x_{i+1}-\textbf x$ (for ease of
calculation we compute $D\textbf e_{i+1}$ first, and then multiply each
side by $D^{-1}$):

$$
D \textbf x = -(L + U)\textbf x + \textbf b \;\;\; and  \;\;\;
D \textbf x_{i+1} = -(L + U)\textbf x_i + \textbf b\\
D\textbf e_{i+1} = D \textbf x_{i} - D\textbf x = -(L+U)\textbf x_{i} + (L+U)\textbf x = -(L+U)\textbf e_{i}\\
\textbf e_{i+1} = -D^{-1}(L+U)\textbf e_i
$$

Whether or not $\textbf e_{i+1}$ increases or decreases depends on the
matrix $-D^{-1}(L+U)$, called the iteration matrix. More specifically,
it depends on the eigenvalues of this iteration matrix.

If we denote the iteration matrix $-D^{-1}(L+U)$ as $T$, we can express
$\textbf e_{i+1}$ as $\textbf e_{i+1} = T\textbf e_i$

Then: $$\textbf e_{i+1} = T^{i+1}\textbf e_0$$

where $\textbf e_0$ is the initial error $\textbf x_{1}-\textbf x$. For
every iteration, the iteration matrix $T$ is acting on the initial error
$\textbf e_0$, and we want this action to be one that decreases the
error.

In order for $\textbf e_{i+1}$ to decrease, *all* eigenvalues of the
iteration matrix $T$ must have magnitudes strictly less than 1, or in
other words, the spectral radius of $T$ must be less than 1 (denoted
$\rho(T) < 1$). With this in mind, it's also useful to realise that this
spectral radius also dictates the rate of convergence - the closer it is
to 0, the faster the iterations will approach the true answer. If
$\rho(T) = 0.5$, for example, we would see the error halve for every
iteration (this would also impact the runtime, if a an error threshold
is specified).

Additionally, according to the interwebs, another *sufficient* condition
for the convergence of the Jacobi Method is if the matrix $A$ is
strictly diagonally dominant, meaning that the magnitude of the diagonal
entry in each row is strictly greater than the sum of the magnitudes of
the other entries in the row. This condition guarantees convergence of
the Jacobi Method, regardless of what your initial guess was. Because
the spectral radius of the iteration matrix isn't always convenient to
find, this is a useful and more inspectable alternative (in that it
applies to the initial matrix $A$ rather than the iteration matrix, and
is also easy to see immediately). Generally, it would probably be better
to first determine if $A$ is strictly diagnonally dominant, and if not
(even after row interchanges), then turn to the spectral radius.

> (b) How long should I run the Jacobi method?

The simplest criterion is to stop (`break`) the iterative method when
the change from iteration `i` to iteration `i + 1` is within a
user-specified margin of error $\epsilon\,$(when no change is observed
between the two successive iterations when rounded to a certain number
of decimal points). This is based on the idea that for an iterative
method, we should see that the change from one iteration to the next
results in smaller and smaller differences, since it is converging
towards the real answer. Or somewhat similarly, we can stop when the
residual $||\textbf b-A\textbf x_i||$ is less than $\epsilon$. Both of
these are probably not the most ideal stopping criterion, but they are
the only ones within my scope of knowledge.

Otherwise, we should specify a certain iteration limit, because in the
worst-case scenario, the method will not converge for the given matrix
in the first place and would continue iterating infinitely without the
iteration limit.

------------------------------------------------------------------------

#### Some sources I found useful

-   <https://www.iosrjournals.org/iosr-jm/papers/vol2-issue2/D0222023.pdf>
-   <https://www-users.cse.umn.edu/~olver/num_/lni.pdf>
-   <https://www3.nd.edu/~zxu2/acms40390F12/Lec-7.3.pdf>
-   *Iterative Methods for Sparse Linear Systems* by Yousef Saad
