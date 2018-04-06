# CART - Classification and Regression Trees

Since I'm learning some Machine Learning algorithms, I decided to code them to gain better understanding of the algorithms and improve my coding skills. This is my first attempt to code them by scratch, and I decided to start with one I like the most: Classification Trees. 

# Why Classification Trees?

I read about them in Varian's 2014 paper: "Big Data: New Tricks for Econometrics". This was the first algorithm that Varian explains, and he compares it with the Logit Model (Logistic Regression). I was amazed by the possibilities this algorithm brings, and I wanted to take all the advantage from it. 

I also saw what libraries such as Sci-Kit Learn have to offer, but they do not allow the user for this two unique features of the CARTs I really like: the way I handles unordered categorical data and the way it is able to use missing information for some features. 

In this approach I'm trying to include them.

# What have I accomplished yet?

- I created a class that decides the best split overall. It returns its Gini Index, the decision rule, the observations that follow the rule and the ones that don't, and many more attributes. 
- This class handles categorical data as it is described in the books: $2^{q-1}-1$ possible combinations. 

# What follows next?

- I need to code the surrogate method for splitting. 
- Make the tree grow. 

