import numpy as np
from numpy import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=200)
np.random.seed(21)

# Aim: to code up row reduction with partial pivoting (choosing the pivot
# to be the largest in absolute magnitude in the pivot search) for an
# m × n matrix

randomMatrix = random.rand(100, 10)
randomIntMat = np.array(random.randint(25, size=(4, 4)),
                        dtype=np.float64)


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
        # not sure why negative zeros exist but this changes them to normal zeros
        # mat[mat == -0] = 0
    return mat


def main():
    start = timer()
    finalMatrix = echelon(randomIntMat)
    end = timer()
    print("FINAL matrix in echelon form, WITH partial pivoting:\n", finalMatrix)
    print(f"Runtime: {end - start} seconds")


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
    return runtimes


# testingRuntimes()
main()

# as a little experiment, we want to see if there are certain values that the runtime increase factor converges towards
def maybeConvergence():
    fiveTo50 = []
    fiftyTo500 = []
    for i in range(10):
        runtimes = testingRuntimes()
        fiveTo50.append(runtimes[1] / runtimes[0])
        fiftyTo500.append(runtimes[2] / runtimes[1])
    plt.subplot(1, 2, 1)
    plt.hist(fiveTo50)
    plt.title("Distribution of increase in runtime from n = 5 to n = 50")
    plt.subplot(1, 2, 2)
    plt.hist(fiftyTo500)
    plt.title("Distribution of increase in runtime from n = 50 to n = 500")
    plt.show()
    return fiveTo50, fiftyTo500

# maybeConvergence()

