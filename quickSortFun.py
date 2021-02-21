
def quicksort(exampleList, firstIndex, lastIndex):
    if(lastIndex > firstIndex):
        pivot = partition(exampleList, firstIndex, lastIndex)

        quicksort(exampleList, firstIndex, lastIndex - 1) #sort to the left of the pivot
                                                          #check and see if values left of pivot
                                                          #are indeed less

        quicksort(exampleList, pivot + 1, lastIndex)     #sort to the right of the pivot
                                                         #check and see if values right of pivot
                                                         #are indeed greater


def partition(exampleList, firstIndex, lastIndex):
    pivot = exampleList[firstIndex]



    sortedList = [listItem for listItem in exampleList if exampleList[listItem] < pivot]

if __name__ == '__main__':
    partition([2, 10, 7, 1, 9], 0, 4)


#Partition:
    #start from leftmost element, keep track of smaller or equal elements of i
    #Find smaller element? Swap with arr[i]