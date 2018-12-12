import sys

def training_set(filename1, filename2):
	fin1 = open(filename1)
	lines_promoter = fin1.readlines()
	fin2 = open(filename2)
	lines_enhancer = fin2.readlines()

	#Note:the output file should be named according to the given input names.
	fout1 = open('sequences.fa','w')
	fout2 = open('label.txt','w')
	for i in range(1, len(lines_promoter)):
		line1 = lines_promoter[i].strip().split(',')
		line2 = lines_enhancer[i].strip().split(',')
		if 'N' in line1[-3] or 'N' in line2[-3]:
			continue
		if line1[-2] == '1':
			fout1.write('>'+'_'.join([line1[0],line1[3],line1[4]])+','+'_'.join([line2[0],line2[3],line2[4]])+'\n')
			fout1.write(line1[-3]+line2[-3]+'\n')
			fout2.write('_'.join([line1[0],line1[3],line1[4]])+','+'_'.join([line2[0],line2[3],line2[4]])+'\t1\n')
		else:
			count2 += 1
			fout1.write('>'+'_'.join([line1[0],line1[3],line1[4]])+','+'_'.join([line2[0],line2[3],line2[4]])+'\n')
			fout1.write(line1[-3]+line2[-3]+'\n')
			fout2.write('_'.join([line1[0],line1[3],line1[4]])+','+'_'.join([line2[0],line2[3],line2[4]])+'\t0\n')
	fout1.close()
	fout2.close()

##MAIN
if sys.argv[1]=='augment':
	#for Basset
	sequences('region1.augment.csv', 'region2.augment.csv')
else:
	#for ls-gkm
	sequences('region1.pos.csv', 'region2.pos.csv')
	sequences('region1.neg.csv', 'region2.neg.csv')

