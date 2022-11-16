import numpy as np

def main():
    while True: 
        print("===========================================================")
        inputMatrix, nodesNumber, branchesNumber = getInputMatrix()
        if validateInputMatrix(inputMatrix): 
            print("The matrix Elements values are not accepted\n")
            continue

        cMatrix = calcCMatrix(inputMatrix, nodesNumber)
        print("\n C Matrix: \n", cMatrix)
        bMatrix = calcBMatrix(inputMatrix, nodesNumber)
        print("\n B Matrix: \n", bMatrix)

        ZbMatrix = getZbMatrix(branchesNumber)
        EbMatrix = getEbMatrix(branchesNumber)
        IbMatrix = getIbMatrix(branchesNumber)
        solveTheEquation(ZbMatrix, bMatrix, EbMatrix, IbMatrix)

def solveTheEquation(zbMatrix, bMatrix, ebMatrix, ibMatrix):
    # Zb * B transpose
    zbTimesBTranspose = np.matmul(zbMatrix, bMatrix.transpose())
    # B * Zb * B transpose (Left side of equation)
    leftSideOfEquation = np.matmul(bMatrix, zbTimesBTranspose)
# =============================================================================
    # B * Eb
    bTimesEb = np.matmul(bMatrix, ebMatrix)

    # Ib * Zb
    ibTimesZb = np.matmul(zbMatrix, ibMatrix)

    # B * Ib * Zb
    bTimesIbTimesZb = np.matmul(bMatrix, ibTimesZb)

    # B * Eb - B * Zb * Ib (Right side of equation)
    rightSideOfEquation = bTimesEb - bTimesIbTimesZb

    # Current In Each Link = solution of the equation
    Il = np.linalg.solve(leftSideOfEquation, rightSideOfEquation)
    print("\n Current On Each Link (loop current) Il: \n", Il)

    # current in each branch = B transpose * IL
    IbInEachBranch = np.matmul(bMatrix.transpose(), Il)
    print("\n Current On Each Branch Ib: \n", IbInEachBranch)

    # Voltage in each branch = (Zb * (IbInEachBranch + IbMatrix) ) - ebMatrix
    VbInEachBranch = np.matmul(zbMatrix, (IbInEachBranch + ibMatrix)) - ebMatrix
    print("\n Voltage On Each Branch Vb: \n", VbInEachBranch)

def getEbMatrix(branchesNumber):
    print("Enter the elements of VOLTAGE SOURCE matrix Eb ORDERED in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    return np.array(entries).reshape(branchesNumber, 1)

def getIbMatrix(branchesNumber):
    print("Enter the elements of CURRENT SOURCE matrix Ib ORDERED in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    return np.array(entries).reshape(branchesNumber, 1)

def getZbMatrix(branchesNumber):
    print("Enter the elements of resistance matrix Zb ORDERED in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    return np.array(entries).reshape(branchesNumber, branchesNumber)

def getInputMatrix():
    nodesNumber = int(input("Enter the number of nodes: "))
    branchesNumber = int(input("Enter the number of branches: "))

    print("Enter the elements of A matrix ORDERED in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    return np.array(entries).reshape(nodesNumber, branchesNumber), nodesNumber, branchesNumber

def validateInputMatrix(inputMatrix):
    column_sums = inputMatrix.sum(axis=0)
    acceptedElementsValues = [0, 1, -1]
    for i in range(len(column_sums)):
        if column_sums[i] not in acceptedElementsValues:
            return True
    return False

def completeAMatrix(inputMatrix):
    column_sums = inputMatrix.sum(axis=0)
    facingMatrix = []
    if any(column_sums[i] != 0 for i in column_sums):
        for i in range(len(column_sums)):
            if column_sums[i] == 0:
                facingMatrix.append(0)
            elif column_sums[i] == 1:
                facingMatrix.append(-1)
            else:
                facingMatrix.append(1)
    A = np.concatenate((inputMatrix, np.array(facingMatrix).reshape(1, len(facingMatrix))), axis=0)
    if len(facingMatrix) > 0:
        return A, True
    return A, False

def getATree(AMatrix, branches, isRowAdded):
    if isRowAdded:
        AMatrix = AMatrix[:-1]
    return AMatrix[:, :branches]

def getInverseOfATree(ATree):
    return np.linalg.inv(ATree)

def getALinks(AMatrix, branches, isRowAdded):
    if isRowAdded:
        AMatrix = AMatrix[:-1]
    return AMatrix[:, branches:]

def calcCMatrix(inputMatrix, branches):
    AMatrix, isRowAdded = completeAMatrix(inputMatrix)
    aTree = getATree(AMatrix, branches, isRowAdded)
    aTreeInverse = getInverseOfATree(aTree)
    aLinks = getALinks(AMatrix, branches, isRowAdded)
    cLinks = getCLinks(aTreeInverse, aLinks)
    cLinksMatrixRows = np.shape(aTree)[0]
    # identity matrix * cLinks matrix
    return np.concatenate((np.identity(cLinksMatrixRows), cLinks), axis=1)

def getCLinks(aTreeInverse, aLinks):
    return np.matmul(aTreeInverse, aLinks)

def getBTree(cLinks):
    return cLinks.transpose()

def calcBMatrix(inputMatrix, branches):
     AMatrix, isRowAdded = completeAMatrix(inputMatrix)
     aTree = getATree(AMatrix, branches, isRowAdded)
     aTreeInverse = getInverseOfATree(aTree)
     aLinks = getALinks(AMatrix, branches, isRowAdded)
     cLinks = getCLinks(aTreeInverse, aLinks)
     bTree = getBTree(cLinks)
     bTreeMatrixRows = np.shape(bTree)[0]
     return np.concatenate((bTree, np.identity(bTreeMatrixRows)), axis=1)

if __name__ == "__main__":
    main()
