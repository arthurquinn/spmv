
if [ $# -ne 1 ]
	then echo "Usage: $0 [vector_size]"
	exit
fi

echo $1

awk -v min=0.1 -v max=3.5 -v num=$1 'BEGIN{ srand(); for (i = 1; i <= num; i++)   print (min+rand()*(max-min+1))}'
