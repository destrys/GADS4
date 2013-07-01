
BEGIN {FS = "," ; OFS=","}
NF == 7  {print $7,$2,$3,7}
NF >= 6  {print $6,$2,$3,6}
NF >= 5  {print $5,$2,$3,5} 
NF >= 4  {print $4,$2,$3,4}
NF >= 3  {print $3,$2,$3,3}
NF >= 2  {print $2,$2,"undef",2}
