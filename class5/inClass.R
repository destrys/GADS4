beer <- read.csv('http://www-958.ibm.com/software/analytics/manyeyes/datasets/af-er-beer-dataset/versions/1.txt', header=TRUE, sep='\t')

beer$good <- beer$WR > 4.36

train.idx <- sample(1:nrow(beer),0.7*nrow(beer))

beer$IPA <- grepl("IPA",beer$Type)
beer$Stout <- grepl("Stout",beer$Type)
beer$Lager <- grepl("Lager",beer$Type)
beer$Ale <- grepl("Ale",beer$Type)
beer$Belgian <- grepl("Belgian",beer$Type)

training <- beer[train.idx,]
test <- beer[-train.idx,]

model <- glm(good ~ IPA + Stout + Lager + Belgian + Ale,data = training,family = 'binomial')

summary(model)

test.predict <- predict(model, test, type = 'response')

test.labels <- test.predict > 0.5

test.truth <- test$WR > 4.36

accuracy <- function(x,y) {
  sum(x == y,na.rm=T)/length(x)
}
head(test.predict)

install.packages('ROCR')
library(ROCR)

test.fix <- merge
pred <- prediction(test.labels,test.truth)