:authors:

Golodyayev Arseniy:
  – MIPT, Skoltech
  – golodyaev.aa@gmail.com
  
Egor Baryshnikov:
  – Skoltech
  – bar.e.s@icloud.com
  
 
:init:

The story of one model that generates kitchen recipes.


:description:

  – /model_generatin_example.ipynb/ jupyter notebook which recipe generation
  
  – /ingredients_sematic.ipynb/ word2vec and ingredients correlation story
  
  – /topic_modeling.ipynb/ recipe clasterization with NLP

_pizza_net_ READ.ME


Neural networks are gradually conquering the world, but not as a Skynet, but as a powerful tool. There are many uses for this marvelous data mining architecture, many of which are not obvious. Below we will talk about one of these applications. Pizza Recipe Generator.

We perfectly understand that there is no limit to perfection, and you can always find solutions better, we will be glad if this happens, but now another story.


there is no machine learning without data.
DATASET:

really hard part of whole project. because of big variation in ingredients name and their bigrams and trigrams. for example pepper we found not only this word in recipes. 
– red pepper 
– black pepper 
– red bell pepper 
– green bell pepper
– chile pepper, and many more
it was not main problem, because sometimes we found this:
[chili,chilli,chiles,chillis] we understand that all mistakes is chile, but if you extrapolate this mistake for all ingredients you will find, that recipe contain set of ingredients with volume more than 100000 unique. It’s not good.

We clear it accurately and receive set of ingredients with depth little bit more than 1000. 
Whole clear DATASET contain more than 300 000 recipes from different countries. Not each but, more of them contain some information about product preprocessing, quantity of each ingredient and some small description.


To see more deeper on data we can plot country cuisine histogram (most recipes from North America)

And some ingredients map for some cousines. Condiments is most popular for each one.

And PIZZA. 


As you can see is not so popular(not quite correctly expressed, it is popular, but limited in recipes)

So next part will solve this problem.


Pizza usually contain very limited set of ingredients, cheese and smth else. Let’s change it. 

Firstly.
If we analyze whole our dataset with magic NLP technic - WORD2VEC, we will find really cool corelation:

This map show most closed word in semantic space to word cheese, seems good. But it’s not all. Some MIT guys create a table of MOLECULAR correlation between many ingredients, thanks them. 

Whole cheeses places in one semantic group, it will help us.


Is it really semantic problem?

To understand is this problem can be solved semantically we us use NLTK algorithm for text topic modeling. we try to clasterize whole recipes dataset into semantic groups. 

Claster number was chosen equal to 15 and here some  visual representation of result. But if we put some real labels into this model.


This histogram show that, desert(blues) will perfectly represent topic 0 and 1. And other real classes have pics in topic where other group is lower.


So we can say that real recipe groups can be divided into clusters or groups semantically quete perfect.

DEEP LEARNING:

We assumed that in whole recipe space exists semi space which represent pizza or other classes.

?/////picture with spaces/semispaces / functional and smt cool///




So our purpose is to find this space and semi space. This task is so close to task of image autoencoding. Where picture can be represented in latent space, which represent big amount of information about this pic. We choose this approach, we want to reproduce each recipe as vector in some latent space and found semi space of Pizza recipes. 

All recipes contains different number of ingredients (from 2,3 up to 20-30) so as we think the best model was that, which can “eat” different number of parameters in one step - RNN. 

We suggested that as in VAE we will fount latent space between two RNN models, where  recipes will grouped.

We chose latent space dimension as 50d vector and start learning. 



In this picture you can see clusterization of some recipe group in latent space. And latent space with decreased dimensionality. Seems as good clasterization%)

But it’s not all. If you remember MIT pepper they write big article about molecular correlation between ingredients. And we use their result for our approximation. If NN put for example mozarella and chicken we will move next ingredients probabilities in side of best correlation with [chicken and mozarella]



Evaluating:

Most of recieved recipes 
pizza:


water, milk, cream_cheese, cheese, olive_oil, chicken, pepper, mozzarella_cheese, pepper, 


pizza:
tomato, egg, milk, olive_oil, olive, garlic, cheese, mozzarella_cheese, black_pepper, chicken, 


pizza:
turkey, parmesan_cheese, cheese, mozzarella_cheese, oil, garlic, tomato, black_pepper, 

Pretty conservative but it’s good result for model. It means that it correctly find pizza semispace. but if we generate more, we will find interesting results……
