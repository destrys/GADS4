###
# Read in Titanic Training Set
###

titanic = read.csv('data/train.csv',head = TRUE)

## Create a Variety of plots to play with this dataset
## Using ggplot

genderbars <- ggplot(titanic,aes(x = sex)) + geom_bar()
genderbars

## Now with who survived
genderbars <- ggplot(titanic,aes(sex,fill = factor(survived))) + geom_bar(position = 'stack')
genderbars

## Add new factor of 'Lived' (survived = 1) and 'Died' (survived = 0)