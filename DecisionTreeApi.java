/**
 * 
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * @author Vishwaksen
 *
 */
class DataSetTestCase {
	List<String> attributes;

	public DataSetTestCase(String testCaseAsString) {
		String[] attributes = testCaseAsString.split("\\s");
		this.attributes = Arrays.asList(attributes);
	}

	public DataSetTestCase(String[] testCaseAsArray) {
		this.attributes = Arrays.asList(testCaseAsArray);
	}
}

public class DecisionTreeApi {

	static class GainInfo {
		double infoGain;
		boolean isDiscrete;
		List<Set<DataSetTestCase>> testCases;
		double threshold;

		public GainInfo(double infoGain, List<Set<DataSetTestCase>> testCases, boolean isDiscrete, double threshold) {
			this.infoGain = infoGain;
			this.testCases = testCases;
			this.isDiscrete = isDiscrete;
			this.threshold = threshold;
		}
	}

	static class DTC4Point5Node {
		boolean isLeaf;
		double threshold; // only present for continuous data
		double classificationError;
		// number of children for continuous attribute will be 2 and for
		// discrete, it will be equal to the number of discrete values
		List<DTC4Point5Node> children = new ArrayList<>();
		boolean isDiscrete;
		boolean isContinuous;
		String classValue;
		int attributeIndex;// index of the attribute considered for this node
		String value;
	}

	// referred the paper on Efficient c4.5
	// http://idb.csie.ncku.edu.tw/tsengsm/course/dm/Paper/ec45.pdf
	static class DecisionTreeC4Point5 {

		private Random random = new Random();
		// private DTC4Point5Node dtRoot;
		public int numberOfAttributes;

		private Map<Integer, Integer> consideredIndexAsPerLevel = new HashMap<>();

		public DTC4Point5Node buildDT(Set<DataSetTestCase> testCasesToBeProcessed, int level, boolean isRandom, int m) {
			boolean allBelongToSameClass = true;
			Iterator<DataSetTestCase> tempIterator = testCasesToBeProcessed.iterator();
			// base case -- if all test cases belong to the same class
			String prevClassFound = "";
			while (tempIterator.hasNext()) {
				if (prevClassFound.equals("")) {
					prevClassFound = tempIterator.next().attributes.get(numberOfAttributes - 1);
				} else {
					if (!prevClassFound.equals(tempIterator.next().attributes.get(numberOfAttributes - 1))) {
						allBelongToSameClass = false;
						break;
					}
				}
			}
			// create a decision node N;
			DTC4Point5Node newDTNode = new DTC4Point5Node();
			if (allBelongToSameClass) {

				newDTNode.classValue = prevClassFound;
				newDTNode.isLeaf = true;
				newDTNode.attributeIndex = -1;
				return newDTNode;
			}

			GainInfo highestInformationGain = null;
			int matchedAttributeIndex = -1;
			if (isRandom) {
				// consider only a subset of attributes; m in this case
				int counter = m;
				while (counter-- > 0) {
					int randomIndex = random.nextInt(numberOfAttributes);
					if (!consideredIndexAsPerLevel.containsKey(randomIndex)
							|| consideredIndexAsPerLevel.get(randomIndex) >= level) {
						GainInfo tempGain = informationGain(randomIndex, testCasesToBeProcessed);
						if (highestInformationGain == null
								|| (tempGain != null && tempGain.infoGain > highestInformationGain.infoGain)) {
							highestInformationGain = tempGain;
							matchedAttributeIndex = randomIndex;
						}
					}
				}
			} else {
				// consider all attributes
				for (int i = 0; i < numberOfAttributes - 1; i++) {
					if (!consideredIndexAsPerLevel.containsKey(i) || consideredIndexAsPerLevel.get(i) >= level) {
						GainInfo tempGain = informationGain(i, testCasesToBeProcessed);
						if (highestInformationGain == null
								|| (tempGain != null && tempGain.infoGain > highestInformationGain.infoGain)) {
							highestInformationGain = tempGain;
							matchedAttributeIndex = i;
						}
					}
				}
			}
			consideredIndexAsPerLevel.put(matchedAttributeIndex, level);
			if (highestInformationGain != null) {
				if (!highestInformationGain.isDiscrete) {
					// for continuous we need to find the threshold using linear
					// search
					double threshold = Integer.MIN_VALUE;
					Iterator<DataSetTestCase> iteratorForThreshold = testCasesToBeProcessed.iterator();
					while (iteratorForThreshold.hasNext()) {
						double currThreshold = Double
								.valueOf(iteratorForThreshold.next().attributes.get(matchedAttributeIndex));
						if (currThreshold > threshold && currThreshold < highestInformationGain.threshold) {
							threshold = currThreshold;
						}
					}
					// minimum threshold fix
					if (threshold == Integer.MIN_VALUE) {
						threshold = highestInformationGain.threshold;
					}
					newDTNode.isDiscrete = false;
					newDTNode.isContinuous = true;
					newDTNode.threshold = threshold;
				} else {
					newDTNode.isDiscrete = true;
					newDTNode.isContinuous = false;
				}

				List<DTC4Point5Node> children = new ArrayList<>();
				for (Set<DataSetTestCase> eachSplit : highestInformationGain.testCases) {
					if (eachSplit.size() == 0) {
						// TODO think about this - no children present
					} else {
						// recurse over each split
						DTC4Point5Node child = buildDT(eachSplit, level + 1, false, -1);
						if (newDTNode.isDiscrete) {
							child.value = eachSplit.iterator().next().attributes.get(matchedAttributeIndex);
						}
						children.add(child);
					}
				}
				newDTNode.children = children;
			}
			// set the attribute number
			newDTNode.attributeIndex = matchedAttributeIndex;
			return newDTNode;
		}

		private GainInfo informationGain(final int i, Set<DataSetTestCase> testCases) {
			// check if attribute is discrete (STRING) or continuous (REAL
			// NUMBER)
			boolean isString = false;
			try {
				// This will raise an exception if the string is not a real
				// number
				if (Double.valueOf(testCases.iterator().next().attributes.get(i).trim()) % 1 == 0) {
					// if integer
					isString = true;
				}
			} catch (Exception e) {
				isString = true;
			}
			if (isString) {
				// Discrete; find all discrete attributes
				Map<String, Set<DataSetTestCase>> allDiscreteAttr = new HashMap<>();
				Iterator<DataSetTestCase> iterator = testCases.iterator();
				while (iterator.hasNext()) {
					DataSetTestCase curr = iterator.next();
					String currAttribute = curr.attributes.get(i);
					if (allDiscreteAttr.containsKey(currAttribute)) {
						allDiscreteAttr.get(currAttribute).add(curr);
					} else {
						Set<DataSetTestCase> newSet = new HashSet<DataSetTestCase>();
						newSet.add(curr);
						allDiscreteAttr.put(currAttribute, newSet);
					}
				}
				// For now considering the relative entropies as 0
				return new GainInfo(
						entropy(new ArrayList<Set<DataSetTestCase>>(allDiscreteAttr.values()), testCases.size()),
						new ArrayList<Set<DataSetTestCase>>(allDiscreteAttr.values()), true, 0.0);
			} else {
				// continuous
				// sorting as per the current attribute values
				List<DataSetTestCase> sortedTestCases = new ArrayList<DataSetTestCase>(testCases);
				Collections.sort(sortedTestCases, new Comparator<DataSetTestCase>() {
					public int compare(DataSetTestCase case1, DataSetTestCase case2) {
						return Double.compare(Double.valueOf(case1.attributes.get(i)),
								Double.valueOf(case2.attributes.get(i)));
					}
				});
				Iterator<DataSetTestCase> sortedIterator = sortedTestCases.iterator();
				double maxGain = Integer.MIN_VALUE;
				GainInfo resultGainInfo = null;
				if (sortedIterator.hasNext()) {
					DataSetTestCase v1 = sortedIterator.next();
					while (sortedIterator.hasNext()) {
						DataSetTestCase v2 = sortedIterator.next();
						double v = (Double.valueOf(v1.attributes.get(i)) + Double.valueOf(v2.attributes.get(i))) / 2.0;
						Set<DataSetTestCase> lowerSet = new HashSet<DataSetTestCase>();
						Set<DataSetTestCase> upperSet = new HashSet<DataSetTestCase>();
						Iterator<DataSetTestCase> sortedIteratorDuplicate = sortedTestCases.iterator();
						while (sortedIteratorDuplicate.hasNext()) {
							DataSetTestCase currElem = sortedIteratorDuplicate.next();
							if (Double.valueOf(currElem.attributes.get(i)) <= v) {
								lowerSet.add(currElem);
							} else {
								upperSet.add(currElem);
							}
						}
						// calculate the gain for this division
						List<Set<DataSetTestCase>> tempList = new ArrayList<>();
						tempList.add(lowerSet);
						tempList.add(upperSet);
						double currGain = entropy(tempList, sortedTestCases.size());
						if (currGain > maxGain) {
							maxGain = currGain;
							resultGainInfo = new GainInfo(currGain, tempList, false, v);
						}
						v1 = v2;
					}
				}
				return resultGainInfo;
			}
		}

		// the whole set entropy is constant, so omitting it from the
		// calculations
		private double entropy(List<Set<DataSetTestCase>> testCasesWithClasses, int total) {
			double result = 0.0f;
			for (Set<DataSetTestCase> eachSet : testCasesWithClasses) {
				double relativeFreq = ((double) eachSet.size() / total);
				result += relativeFreq * Math.log(relativeFreq) / Math.log(2);
			}
			return result * -1;
		}
	}

	static Set<DataSetTestCase> dataAsSet = new HashSet<>();
	private static String DATA_1_PATH = "B:\\fall16\\601\\project3\\project3_dataset1.txt";
	private static String DATA_2_PATH = "B:\\fall16\\601\\project3\\project3_dataset2.txt";
	private static String DATA_2_TEST_PATH = "B:\\fall16\\601\\project3\\project3_dataset2_test.txt";
	private static String DATA_4_PATH = "B:\\fall16\\601\\project3\\project3_dataset4.txt";
	private static String EXAMPLE_PATH = "B:\\fall16\\601\\project3\\example.txt";
	private static Random random = new Random();

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("With Cross Validation: ");
		withCrossValidation();
		System.out.println("Without Cross Validation: ");
		withoutCrossValidation();

	}

	private static void withCrossValidation() {

		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_4_PATH)))) {
			int folds = 10;
			List<DataSetTestCase> allTestCases = new ArrayList<>();
			String currLine = null;
			int numberOfAttributes = 0;
			while ((currLine = br.readLine()) != null) {
				String[] split = currLine.split("\\s");
				allTestCases.add(new DataSetTestCase(currLine));
				numberOfAttributes = split.length;
			}

			int limit = allTestCases.size() / folds;
			double finalAcc = 0;
			double finalPrec = 0;
			double finalRec = 0;
			double finalF = 0;

			for (int f = 0; f < folds; f++) {
				Set<DataSetTestCase> validationDataSet = new HashSet<>();
				dataAsSet.clear();
				for (int count = 0; count < f * limit; count++) {
					dataAsSet.add(allTestCases.get(count));
				}
				for (int count = f * limit; count < f * limit + limit; count++) {
					validationDataSet.add(allTestCases.get(count));
				}
				for (int count = f * limit + limit; count < allTestCases.size(); count++) {
					dataAsSet.add(allTestCases.get(count));
				}

				// building the tree
				DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
				tree.numberOfAttributes = numberOfAttributes;
				DTC4Point5Node root = tree.buildDT(dataAsSet, 0, false, -1);

				// printing the decision tree using BFS
				System.out.println("Decision Tree - BFS");
				LinkedList<DTC4Point5Node> queue = new LinkedList<>();
				queue.offer(root);
				int stepsToLevelChange = 1;
				int displayCounter = 0;
				while (!queue.isEmpty()) {
					DTC4Point5Node curr = queue.poll();
					stepsToLevelChange--;
					System.out.print(displayCounter++ + "\t");
					if (!curr.isLeaf) {
						for (DTC4Point5Node eachChild : curr.children) {
							queue.offer(eachChild);
						}
					}
					if (stepsToLevelChange == 0) {
						stepsToLevelChange = queue.size();
						System.out.println();
					}
				}

				// validation using DFS
				Map<DataSetTestCase, String> results = validation(validationDataSet, root);
				int tn = 0;
				int tp = 0;
				int fp = 0;
				int fn = 0;
				for (Map.Entry<DataSetTestCase, String> eachResult : results.entrySet()) {
					DataSetTestCase key = eachResult.getKey();
					String value = eachResult.getValue().trim();
					if (value.equals("0")) {
						// predicted 0
						if (value.equals(key.attributes.get(numberOfAttributes - 1))) {
							tn++;
						} else {
							fn++;
						}
					} else {
						// predicted 1
						if (value.equals(key.attributes.get(numberOfAttributes - 1))) {
							tp++;
						} else {
							fp++;
						}
					}
				}

				finalAcc += (tp + tn == 0) ? 0 : 100.0 * ((double) (tp + tn) / (tn + tp + fn + fp));
				finalPrec += (tp == 0) ? 0 : (double) tp / (tp + fp);
				finalRec += (tp == 0) ? 0 : (double) tp / (tp + fn);
				finalF += (tp == 0) ? 0 : (2.0 * tp) / ((2.0 * tp) + fn + fp);
				// TODO do multiple dry runs
			}

			// display the measure
			finalAcc /= folds;
			finalPrec /= folds;
			finalRec /= folds;
			finalF /= folds;

			System.out.println("Accuracy: " + finalAcc);
			System.out.println("Precision: " + finalPrec);
			System.out.println("Recall: " + finalRec);
			System.out.println("F-1 measure: " + finalF);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// returns a Map containing the DataSet test case and the predicted class
	public static Map<DataSetTestCase, String> validation(Set<DataSetTestCase> validationDataSet, DTC4Point5Node root) {
		LinkedList<DTC4Point5Node> stack = new LinkedList<>();
		Map<DataSetTestCase, String> result = new HashMap<>();

		int differencesInOutput = 0;
		int outputCounter = 0;
		for (DataSetTestCase eachValidationNode : validationDataSet) {
			String predictedClass = "";
			stack.clear(); // TODO test this -- like how it affects the results
			stack.push(root);
			while (!stack.isEmpty()) {
				DTC4Point5Node curr = stack.pop();
				if (curr.isLeaf) {
					// System.out.println("Predicted Class: " +
					// curr.classValue);
					predictedClass = curr.classValue;
					// System.out.println("Actual Class: "
					// +
					// eachValidationNode.attributes.get(eachValidationNode.attributes.size()
					// - 1));
					if (!curr.classValue.trim().equals(
							eachValidationNode.attributes.get(eachValidationNode.attributes.size() - 1).trim())) {
						differencesInOutput++;
					}
					outputCounter++;
					// System.out.println();
					break;
				}
				int indexInConsideration = curr.attributeIndex;
				if (curr.isDiscrete) {
					boolean found = false;
					// for discrete -- we find the matching value from the
					// children and add it to the stack
					for (DTC4Point5Node eachChild : curr.children) {
						if (eachChild.value.trim().equals(eachValidationNode.attributes.get(indexInConsideration))) {
							stack.push(eachChild);
							found = true;
							break;
						}
					}
					if (!found) {
						// randomly assign a child
						int randomChild = random.nextInt(curr.children.size());
						stack.push(curr.children.get(randomChild));
					}
				} else if (indexInConsideration != -1 && (curr.children.size() > 0)) {
					// for continuous
					if (Double.valueOf(eachValidationNode.attributes.get(indexInConsideration)) <= curr.threshold) {
						// add left child
						stack.push(curr.children.get(0));
					} else {
						// add right child
						stack.push(curr.children.get(1));
					}
				}
			}
			result.put(eachValidationNode, predictedClass);
		}

		return result;
	}

	private static void withoutCrossValidation() {
		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_4_PATH)))) {
			int folds = 1;
			List<DataSetTestCase> allTestCases = new ArrayList<>();
			String currLine = null;
			int numberOfAttributes = 0;
			while ((currLine = br.readLine()) != null) {
				String[] split = currLine.split("\\s");
				allTestCases.add(new DataSetTestCase(currLine));
				numberOfAttributes = split.length;
			}

			double finalAcc = 0;
			double finalPrec = 0;
			double finalRec = 0;
			double finalF = 0;

			for (int f = 0; f < folds; f++) {
				Set<DataSetTestCase> validationDataSet = new HashSet<>();
				dataAsSet.clear();
				int counter = random.nextInt(3);
				int validation = 0;
				for (int count = 0; count < allTestCases.size(); count++) {
					if (counter == 0 && validation < 20) {
						counter = random.nextInt(3);
						validation++;
						validationDataSet.add(allTestCases.get(count));
					} else {
						counter--;
						dataAsSet.add(allTestCases.get(count));
					}
				}

				// building the tree
				DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
				tree.numberOfAttributes = numberOfAttributes;
				DTC4Point5Node root = tree.buildDT(dataAsSet, 0, false, -1);

				// printing the decision tree using BFS
				System.out.println("Decision Tree - BFS");
				LinkedList<DTC4Point5Node> queue = new LinkedList<>();
				queue.offer(root);
				int stepsToLevelChange = 1;
				int displayCounter = 0;
				while (!queue.isEmpty()) {
					DTC4Point5Node curr = queue.poll();
					stepsToLevelChange--;
					System.out.print(displayCounter++ + "\t");
					if (!curr.isLeaf) {
						for (DTC4Point5Node eachChild : curr.children) {
							queue.offer(eachChild);
						}
					}
					if (stepsToLevelChange == 0) {
						stepsToLevelChange = queue.size();
						System.out.println();
					}
				}

				// validation using DFS
				Map<DataSetTestCase, String> results = validation(validationDataSet, root);
				int tn = 0;
				int tp = 0;
				int fp = 0;
				int fn = 0;
				for (Map.Entry<DataSetTestCase, String> eachResult : results.entrySet()) {
					DataSetTestCase key = eachResult.getKey();
					String value = eachResult.getValue().trim();
					if (value.equals("0")) {
						// predicted 0
						if (value.equals(key.attributes.get(numberOfAttributes - 1))) {
							tn++;
						} else {
							fn++;
						}
					} else {
						// predicted 1
						if (value.equals(key.attributes.get(numberOfAttributes - 1))) {
							tp++;
						} else {
							fp++;
						}
					}
				}

				finalAcc += (tp + tn == 0) ? 0 : 100.0 * ((double) (tp + tn) / (tn + tp + fn + fp));
				finalPrec += (tp == 0) ? 0 : (double) tp / (tp + fp);
				finalRec += (tp == 0) ? 0 : (double) tp / (tp + fn);
				finalF += (tp == 0) ? 0 : (2.0 * tp) / ((2.0 * tp) + fn + fp);
				// TODO do multiple dry runs
			}

			// display the measure
			finalAcc /= folds;
			finalPrec /= folds;
			finalRec /= folds;
			finalF /= folds;

			System.out.println("Accuracy: " + finalAcc);
			System.out.println("Precision: " + finalPrec);
			System.out.println("Recall: " + finalRec);
			System.out.println("F-1 measure: " + finalF);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
