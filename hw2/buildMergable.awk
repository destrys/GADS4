
BEGIN {FS = "," ; OFS=","}
NF == 5  {print $5,$2,$3,5} 
NF >= 4  {print $4,$2,$3,4}
NF >= 3  {print $3,$2,$3,3}
NF >= 2  {print $2,$2,"undef",2}
