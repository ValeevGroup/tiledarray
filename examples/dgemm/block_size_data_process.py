import linecache, os

row_blocks=[]
col_blocks=[]
gflops=[]

for fn in os.listdir('.'):
    if(fn[0] == "o"):
        row_blocks.append(int(linecache.getline(fn,1).split()[0]))
        col_blocks.append(int(linecache.getline(fn,1).split()[1]))
        gflops.append(float(linecache.getline(fn,16).split()[3]))
    
outfile = open("mathout.txt", 'w')
for n in range(len(row_blocks)):
    print("{", row_blocks[n], ",", col_blocks[n], ",", gflops[n], "},", file=outfile)
