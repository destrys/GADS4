setwd('~/Documents/GeneralAssembly/DataScience/GADS4/class2/538model/data/')

census1 <- read.csv('census_data_2000.csv',as.is=TRUE)
View(census1)
str(census1)

census2 <- read.csv('census_demographics.csv',as.is=T)

# How can we tell if the dataframes are different?
# Let's see how many columns match:

#Intersect shows which of names(census2) is in names(census1):
intersect(names(census1),names(census2))

#Use setdiff() to see what is missing:
setdiff(names(census1),names(census2))
setdiff(names(census2),names(census1))

#lets change the capital 'State' to 'state'
names(census1)[names(census1) == 'State'] <- 'state'

# And now the column names are the same!
setdiff(names(census1),names(census2))

# Let's look at what each line is about:
census1$state

# What's the older pop's deal?
sum(census1$older_pop)
sum(census2$older_pop)
sum(census1$vote_pop)
sum(census2$vote_pop)

# Let's play with merge()
census <- merge(census1,census2,by='state')  # FAIL!
head(census1$state)
head(census2$state)
#durp, census2 has all caps, let's fix census1
census1$state <- toupper(census1$state)

census <- merge(census1,census2,by='state') # WIN!
names(census)   # appends .x and .y to names()

# Now to add data...
census1$year <- 2000
census2$year <- 2012

# How to swap column order...
census2 <- census2[,names(census1)]

# Now to combine the dataframes....
censusAll <- rbind(census1,census2)
# Boom! 

# Now to make it sexier...
molten <- melt(censusAll,id.vars=c("state","year"))
# cast() is the opposite of melt()

################
# Now for ggplot2
################

# First we look at base graphics...
plot(census1$vote_pop ~ census1$average_income) # woo, ugle scatterplot

# https://github.com/arahuja/GADS4/wiki/Basic-reference-for-ggplot2

library(ggplot2)
ggplot(data = census1) + aes(x = vote_pop) + geom_histogram()
ggplot(data = census1) + aes(x = vote_pop, y = older_pop) + geom_point() 

ggplot(data = census1) + aes(x = vote_pop) + geom_density()

# MAPS!
#install.packages('maps')
library(maps)