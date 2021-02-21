# This is what a comment looks like
fruits = ['apples','oranges','pears','bananas']
for fruit in fruits:
    print(fruit + ' for sale')
    
fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75}
for fruit, price in fruitPrices.items(): #We mention "fruit, price" in this case
                                        #because in dictionaries, there's key value pairs
                                        #So calling fruitPrices.items() will return both the
                                        #fruit and the price

    if price < 2.00:
          print('%s cost %f a pound' % (fruit, price))

    else:
          print(fruit + ' are too expensive!')


    #print ('%s cost %f a pound' % (fruit, price))


#print(fruitPrices.items())
# This prints the dictionary's key-value pairs in a different format
# While mentioning it's a dictionary