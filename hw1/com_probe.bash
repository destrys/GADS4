# Check Column Headings and first couple lines
head -n3 data/train.csv

# Check number of lines (892 including header)
wc -l data/train.csv

# Check number of Surviving (342)
grep ^1 data/train.csv |wc -l

# Check number of Women (314)
grep female data/train.csv |wc -l

# See if any genders are neither female nor male (none)
grep -v male data/train.csv

