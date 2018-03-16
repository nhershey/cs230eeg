def parseSeizures():
	seizure_file = open("seizures.txt", "r")
	seizure_names = seizure_file.readlines()

	for i in range(len(seizure_names)):
		seizure_names[i] = seizure_names[i].strip("\n")

	seizure_tuples = []
	last_file = seizure_names[0]
	num_seizures_in_file = 0
	seizure_tuples.append((last_file, num_seizures_in_file))
	for i in range(1, len(seizure_names)):
		if (seizure_names[i] == last_file):
			num_seizures_in_file += 1
		else:
			num_seizures_in_file = 0
			last_file = seizure_names[i]
		seizure_tuples.append((last_file, num_seizures_in_file))

	seizure_file = open('seizures_marked.txt', 'w+')
	for name, count in seizure_tuples:
		seizure_file.write("%s,%s\n" % (name, count))

def parseNonseizures():
	non_seizure_file = open("non_seizures.txt", "r")
	non_seizure_names = non_seizure_file.readlines()

	non_seizure_file = open('nonSeizures_marked.txt', 'w+')
	for i in range(len(non_seizure_names)):
		non_seizure_names[i] = non_seizure_names[i].strip("\n")
		non_seizure_file.write("%s,%s\n" % (non_seizure_names[i], -1))

if __name__ == "__main__":
    parseSeizures()
    parseNonseizures()