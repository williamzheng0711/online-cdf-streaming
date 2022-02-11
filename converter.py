import csv

network_trace_dir = './dataset/fyp_lab/'

time_inveral = 3
amount = 0 
timeTrack = 1615285137.282262000

for suffixNum in range(2,3):
    networkThroughput = []
    with open( network_trace_dir+str(suffixNum) + ".txt"  ) as file1:
        for line in file1:
            parse = line.split()
            if (float(parse[0]) - timeTrack ) > time_inveral:
                throughputLast = amount / ( float(parse[0]) - timeTrack  )
                timeTrack = float(parse[0])
                networkThroughput.append( throughputLast )
                amount = 0
            amount = amount + float(parse[1])

with open('2s.csv', 'w') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            for val1 in networkThroughput:
                wr.writerow([val1])


            
