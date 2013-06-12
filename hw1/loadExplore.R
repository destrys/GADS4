###
# Read in Titanic Training Set
###

titanic = read.csv('data/train.csv',head = TRUE)

###
# Data Cleaning?
###

# The data are already in pretty nice shape.

###
# Create a Variety of plots to play with this dataset
# Using ggplot
###

# Bar Plot of Genders
gbars <- ggplot(titanic,aes(x = sex))         # Create gg w/ aes
gbars <- gbars + geom_bar()                   # Add geom
gbars <- gbars + theme_classic(16,"Times")    # I like classis better
gbars <- gbars + xlab('') + ylab('People')    # Fix labels
gbars <- gbars + theme(axis.line = element_blank())  # Ditch the axes
gbars <- gbars + theme(axis.ticks = element_blank()) # And the ticks
gbars

# Print this one to disk
png(filename="gender.png", width=600, height=480)
gbars
dev.off()
genderbars <- ggplot(titanic,aes(x = sex,fill = factor(survived))) + geom_bar()
genderbars

# That's great, but the data is about who survived and who didn't, so let's add
# that to this plot

gbars <- ggplot(titanic,aes(x = sex, fill = factor(survived)))  # Create gg w/ aes
# We don''t really care about the absolute number of men or women, but whether likelihood
# of survival is related to gender, so we'll dodge the bars, not stack them.
gbars <- gbars + geom_bar(position = 'dodge')                   # Add geom 
gbars <- gbars + theme_classic(16,"Times")    # I like classis better
gbars <- gbars + xlab('') + ylab('People')    # Fix labels
gbars <- gbars + theme(axis.line = element_blank())  # Ditch the axes
gbars <- gbars + theme(axis.ticks = element_blank()) # And the ticks
# How do I change the Legend Title and the labels?
# Better yet, can I just label the bars and get rid of the legend?
n_survived <- count(titanic,vars = c("survived", "sex"))
str(n_survived)
n_survived$text[n_survived$survived == 0] <- 'Died'
n_survived$text[n_survived$survived == 1] <- 'Survived'
gbars <- gbars + geom_text(aes(sex,freq+10,label = text),data = n_survived,position=position_dodge(width=0.9))
gbars <- gbars + theme(legend.position = "none")
gbars



## Now with who survived
genderbars <- ggplot(titanic,aes(sex,fill = factor(survived))) + geom_bar(position = 'stack')
genderbars

