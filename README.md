# rnaformer
Kaggle Ribonanza Stanford Competition

This Repo contains the majority of my code for the Stanford Ribonanza Kaggle Competition, where the goal is to use the large amount of RNA sequence data they provide tp predict the reactivity of unseen RNA sequences. I was originally using Kaggle and Colab Notebooks to build out and train my models, but I soon found that to really compete I need better access to some hardware accelerators. To achieve this I went with the Vast.ai cloud computing platform, which was tremendous experience. They provide access to a plethora of different types of GPU's.

This pushed me to streamline my code into python files, properly structure them into appropriate component files, continually train and ultimately save my models. This allowed me to have a far greater number of model iterations and experiments. This is reflective in my models.py file, where there are numerous different architechtures I played around with including the transformer alone, transformer paired with RNN's, F-Nets and more. 
