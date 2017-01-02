/**
 * 
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import project3.DecisionTreeApi.DTC4Point5Node;
import project3.DecisionTreeApi.DecisionTreeC4Point5;

/**
 * @author Vishwaksen
 *
 */
public class Boosting {

	private static String DATA_1_PATH = "B:\\fall16\\601\\project3\\project3_dataset1.txt";
	private static String DATA_2_PATH = "B:\\fall16\\601\\project3\\project3_dataset2.txt";
	private static String DATA_4_PATH = "B:\\fall16\\601\\project3\\project3_dataset4.txt";
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

	public static Map<DataSetTestCase, String> validation(List<DataSetTestCase> validationDataSet, int rounds,
			Set<String> allClasses, List<Double> classifierErrors, List<DTC4Point5Node> classifierModels) {
		Map<DataSetTestCase, String> result = new HashMap<>();

		for (DataSetTestCase eachTestCase : validationDataSet) {
			Map<String, Double> classWeights = new HashMap<>();
			// initialize class weights
			for (String eachClass : allClasses) {
				classWeights.put(eachClass, 0.0);
			}
			int index = 0;
			// iterating through each classifier
			while (index < rounds) {
				double weightForVote = Math.log((1 - classifierErrors.get(index)) / classifierErrors.get(index));
				Set<DataSetTestCase> temp = new HashSet<>();
				temp.add(eachTestCase);
				String predictedClass = DecisionTreeApi.validation(temp, classifierModels.get(index)).get(eachTestCase);
				classWeights.put(predictedClass.trim(), classWeights.get(predictedClass.trim()) + weightForVote);
				index++;
			}
			// find the class with the highest weight
			String predictedClass = "";
			double maxWeight = Integer.MIN_VALUE;
			for (Map.Entry<String, Double> eachClassWeight : classWeights.entrySet()) {
				if (eachClassWeight.getValue() > maxWeight) {
					maxWeight = eachClassWeight.getValue();
					predictedClass = eachClassWeight.getKey();
				}
			}
			result.put(eachTestCase, predictedClass);
		}
		return result;
	}

	private static void withCrossValidation() {

		int rounds = 6;
		int folds = 10;

		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_2_PATH)))) {
			List<DataSetTestCase> allTestCases = new ArrayList<>();
			String currLine = null;
			int numberOfAttributes = 0;
			Set<String> allClasses = new HashSet<>();

			while ((currLine = br.readLine()) != null) {
				String[] split = currLine.split("\\s");
				allTestCases.add(new DataSetTestCase(currLine));
				numberOfAttributes = split.length;
				if (!allClasses.contains(split[numberOfAttributes - 1])) {
					allClasses.add(split[numberOfAttributes - 1]);
				}
			}

			int limit = allTestCases.size() / folds;
			double finalAcc = 0;
			double finalPrec = 0;
			double finalRec = 0;
			double finalF = 0;

			for (int f = 0; f < folds; f++) {
				List<DataSetTestCase> trainingDataSet = new ArrayList<>();
				List<DataSetTestCase> validationDataSet = new ArrayList<>();

				for (int count = 0; count < f * limit; count++) {
					trainingDataSet.add(allTestCases.get(count));
				}
				for (int count = f * limit; count < f * limit + limit; count++) {
					validationDataSet.add(allTestCases.get(count));
				}
				for (int count = f * limit + limit; count < allTestCases.size(); count++) {
					trainingDataSet.add(allTestCases.get(count));
				}
				int tn = 0;
				int tp = 0;
				int fp = 0;
				int fn = 0;

				int N = trainingDataSet.size();

				// Initial Weights
				Map<DataSetTestCase, Double> weights = new HashMap<>();
				for (DataSetTestCase eachDatum : trainingDataSet) {
					weights.put(eachDatum, 1.0 / N);
				}

				List<Double> classifierErrors = new ArrayList<>();
				List<DTC4Point5Node> classifierModels = new ArrayList<>();

				for (int r = 0; r < rounds; r++) {
					// create bootstrap sample - 63.2% of original records with
					// replacement
					int sampleSize = (int) (0.632 * N);
					DecisionTreeApi.dataAsSet.clear();
					while (DecisionTreeApi.dataAsSet.size() < sampleSize) {
						int nextIndex = random.nextInt(N);
						DecisionTreeApi.dataAsSet.add(trainingDataSet.get(nextIndex));
					}
					// creating the decision tree
					DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
					tree.numberOfAttributes = numberOfAttributes;
					DTC4Point5Node root = tree.buildDT(DecisionTreeApi.dataAsSet, 0, false, -1);
					classifierModels.add(root);

					Map<DataSetTestCase, String> results = DecisionTreeApi
							.validation(new HashSet<DataSetTestCase>(trainingDataSet), root);

					// calculate the error
					double error = 0.0;
					for (Map.Entry<DataSetTestCase, String> eachResult : results.entrySet()) {
						if (!eachResult.getValue().trim()
								.equals(eachResult.getKey().attributes.get(numberOfAttributes - 1).trim())) {
							error += weights.get(eachResult.getKey());
						}
					}
					if (error > 0.5) {
						// sample again
						continue;
					}
					classifierErrors.add(error);

					double sumOfNewWeights = 0;
					// updating the weights
					for (DataSetTestCase eachTestCase : DecisionTreeApi.dataAsSet) {
						double newWeight = weights.get(eachTestCase);
						// update the weight only if correctly classified
						if (results.get(eachTestCase).trim()
								.equals(eachTestCase.attributes.get(numberOfAttributes - 1).trim())) {
							newWeight *= error / (1 - error);
							weights.put(eachTestCase, newWeight);
						}
						sumOfNewWeights += newWeight;
					}
					// normalize the weights
					for (DataSetTestCase eachTestCase : DecisionTreeApi.dataAsSet) {
						weights.put(eachTestCase, (weights.get(eachTestCase) / sumOfNewWeights));
					}
				}

				// employing the ensemble for augmented classification
				Map<DataSetTestCase, String> validationResults = validation(validationDataSet, rounds, allClasses,
						classifierErrors, classifierModels);
				for (Map.Entry<DataSetTestCase, String> eachValidationResult : validationResults.entrySet()) {
					DataSetTestCase key = eachValidationResult.getKey();
					String value = eachValidationResult.getValue().trim();
					if (value.equals("0")) {
						// predicted 0
						if (value.equals(key.attributes.get(numberOfAttributes - 1).trim())) {
							tn++;
						} else {
							fn++;
						}
					} else {
						// predicted 1
						if (value.equals(key.attributes.get(numberOfAttributes - 1).trim())) {
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

	private static void withoutCrossValidation() {

		int rounds = 6;
		int folds = 1;

		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_2_PATH)))) {
			List<DataSetTestCase> allTestCases = new ArrayList<>();
			String currLine = null;
			int numberOfAttributes = 0;
			Set<String> allClasses = new HashSet<>();

			while ((currLine = br.readLine()) != null) {
				String[] split = currLine.split("\\s");
				allTestCases.add(new DataSetTestCase(currLine));
				numberOfAttributes = split.length;
				if (!allClasses.contains(split[numberOfAttributes - 1])) {
					allClasses.add(split[numberOfAttributes - 1]);
				}
			}

			double finalAcc = 0;
			double finalPrec = 0;
			double finalRec = 0;
			double finalF = 0;

			for (int f = 0; f < folds; f++) {
				List<DataSetTestCase> trainingDataSet = new ArrayList<>();
				List<DataSetTestCase> validationDataSet = new ArrayList<>();

				int counter = random.nextInt(3);
				int validation = 0;
				for (int count = 0; count < allTestCases.size(); count++) {
					if (counter == 0 && validation < 20) {
						counter = random.nextInt(3);
						validation++;
						validationDataSet.add(allTestCases.get(count));
					} else {
						counter--;
						trainingDataSet.add(allTestCases.get(count));
					}
				}

				int tn = 0;
				int tp = 0;
				int fp = 0;
				int fn = 0;

				int N = trainingDataSet.size();

				// Initial Weights
				Map<DataSetTestCase, Double> weights = new HashMap<>();
				for (DataSetTestCase eachDatum : trainingDataSet) {
					weights.put(eachDatum, 1.0 / N);
				}

				List<Double> classifierErrors = new ArrayList<>();
				List<DTC4Point5Node> classifierModels = new ArrayList<>();

				for (int r = 0; r < rounds; r++) {
					// create bootstrap sample - 63.2% of original records with
					// replacement
					int sampleSize = (int) (0.632 * N);
					DecisionTreeApi.dataAsSet.clear();
					while (DecisionTreeApi.dataAsSet.size() < sampleSize) {
						int nextIndex = random.nextInt(N);
						DecisionTreeApi.dataAsSet.add(trainingDataSet.get(nextIndex));
					}
					// creating the decision tree
					DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
					tree.numberOfAttributes = numberOfAttributes;
					DTC4Point5Node root = tree.buildDT(DecisionTreeApi.dataAsSet, 0, false, -1);
					classifierModels.add(root);

					Map<DataSetTestCase, String> results = DecisionTreeApi
							.validation(new HashSet<DataSetTestCase>(trainingDataSet), root);

					// calculate the error
					double error = 0.0;
					for (Map.Entry<DataSetTestCase, String> eachResult : results.entrySet()) {
						if (!eachResult.getValue().trim()
								.equals(eachResult.getKey().attributes.get(numberOfAttributes - 1).trim())) {
							error += weights.get(eachResult.getKey());
						}
					}
					if (error > 0.5) {
						// sample again
						continue;
					}
					classifierErrors.add(error);

					double sumOfNewWeights = 0;
					// updating the weights
					for (DataSetTestCase eachTestCase : DecisionTreeApi.dataAsSet) {
						double newWeight = weights.get(eachTestCase);
						// update the weight only if correctly classified
						if (results.get(eachTestCase).trim()
								.equals(eachTestCase.attributes.get(numberOfAttributes - 1).trim())) {
							newWeight *= error / (1 - error);
							weights.put(eachTestCase, newWeight);
						}
						sumOfNewWeights += newWeight;
					}
					// normalize the weights
					for (DataSetTestCase eachTestCase : DecisionTreeApi.dataAsSet) {
						weights.put(eachTestCase, (weights.get(eachTestCase) / sumOfNewWeights));
					}
				}

				// employing the ensemble for augmented classification
				Map<DataSetTestCase, String> validationResults = validation(validationDataSet, rounds, allClasses,
						classifierErrors, classifierModels);
				for (Map.Entry<DataSetTestCase, String> eachValidationResult : validationResults.entrySet()) {
					DataSetTestCase key = eachValidationResult.getKey();
					String value = eachValidationResult.getValue().trim();
					if (value.equals("0")) {
						// predicted 0
						if (value.equals(key.attributes.get(numberOfAttributes - 1).trim())) {
							tn++;
						} else {
							fn++;
						}
					} else {
						// predicted 1
						if (value.equals(key.attributes.get(numberOfAttributes - 1).trim())) {
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
