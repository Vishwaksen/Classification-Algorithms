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

import project3.DecisionTreeApi.DTC4Point5Node;
import project3.DecisionTreeApi.DecisionTreeC4Point5;

/**
 * @author Vishwaksen
 *
 */
public class RandomDecisionForest {

	private static String DATA_1_PATH = "B:\\fall16\\601\\project3\\project3_dataset1.txt";
	private static String DATA_2_PATH = "B:\\fall16\\601\\project3\\project3_dataset2.txt";
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

		// T number of trees
		int T = 6; // test on multiple values
		int folds = 10;

		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_2_PATH)))) {
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
				List<DataSetTestCase> validationDataSet = new ArrayList<>();
				List<DataSetTestCase> trainingDataSet = new ArrayList<>();
				Map<DataSetTestCase, String> finalResultForAFold = new HashMap<>();

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

				// creating samples from the training set
				int N = trainingDataSet.size();
				// select 20% features at each node -- idea for this -- modify
				// the api
				int m = (int) (0.2 * numberOfAttributes);
				// storing results from all T trees
				List<Map<DataSetTestCase, String>> accumulatedResults = new ArrayList<>();

				// for each tree
				for (int t = 0; t < T; t++) {
					// Choose a training set by choosing N times (N is the
					// number of
					// training examples) with replacement from the training set
					// creating N samples for each tree
					DecisionTreeApi.dataAsSet.clear();
					for (int n = 0; n < N; n++) {
						int randomIndex = random.nextInt(N);
						DecisionTreeApi.dataAsSet.add(trainingDataSet.get(randomIndex));
					}

					// building the tree
					DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
					// TODO this won't be required for forest
					tree.numberOfAttributes = numberOfAttributes;
					// to handle the random case
					// true for random selection of m features
					DTC4Point5Node root = tree.buildDT(DecisionTreeApi.dataAsSet, 0, true, m);

					System.out.print("Tree " + t + " ");
					// validation using DFS
					accumulatedResults
							.add(DecisionTreeApi.validation(new HashSet<DataSetTestCase>(validationDataSet), root));
				}

				// choose the tree with the most votes
				for (int v = 0; v < validationDataSet.size(); v++) {
					List<String> votes = new ArrayList<>();
					for (int a = 0; a < accumulatedResults.size(); a++) {
						// for each map find the votes
						votes.add(accumulatedResults.get(a).get(validationDataSet.get(v)));
					}
					// find max vote and add to result
					Map<String, Integer> votesCount = new HashMap<>();
					for (String vote : votes) {
						if (votesCount.containsKey(vote)) {
							votesCount.put(vote, votesCount.get(vote) + 1);
						} else {
							votesCount.put(vote, 1);
						}
					}
					// linear scan to find the highest vote
					int max = Integer.MIN_VALUE;
					String finalVote = "";
					for (Map.Entry<String, Integer> eachMaxVote : votesCount.entrySet()) {
						if (eachMaxVote.getValue() > max) {
							max = eachMaxVote.getValue();
							finalVote = eachMaxVote.getKey();
						}
					}
					finalResultForAFold.put(validationDataSet.get(v), finalVote);
				}

				for (Map.Entry<DataSetTestCase, String> eachFinalResult : finalResultForAFold.entrySet()) {
					DataSetTestCase key = eachFinalResult.getKey();
					String value = eachFinalResult.getValue().trim();
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
				// /TODO remove this
				System.out.println(
						"Final Result Set size: " + finalResultForAFold.size() + " for " + validationDataSet.size());

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

	private static void withoutCrossValidation() {

		// T number of trees
		int T = 6; // test on multiple values
		int folds = 1;

		try (BufferedReader br = new BufferedReader(new FileReader(new File(DATA_1_PATH)))) {
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
				List<DataSetTestCase> validationDataSet = new ArrayList<>();
				List<DataSetTestCase> trainingDataSet = new ArrayList<>();
				Map<DataSetTestCase, String> finalResultForAFold = new HashMap<>();

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

				// creating samples from the training set
				int N = trainingDataSet.size();
				// select 20% features at each node -- idea for this -- modify
				// the api
				int m = (int) (0.2 * numberOfAttributes);
				// storing results from all T trees
				List<Map<DataSetTestCase, String>> accumulatedResults = new ArrayList<>();

				// for each tree
				for (int t = 0; t < T; t++) {
					// Choose a training set by choosing N times (N is the
					// number of
					// training examples) with replacement from the training set
					// creating N samples for each tree
					DecisionTreeApi.dataAsSet.clear();
					for (int n = 0; n < N; n++) {
						int randomIndex = random.nextInt(N);
						DecisionTreeApi.dataAsSet.add(trainingDataSet.get(randomIndex));
					}

					// building the tree
					DecisionTreeC4Point5 tree = new DecisionTreeC4Point5();
					// TODO this won't be required for forest
					tree.numberOfAttributes = numberOfAttributes;
					// to handle the random case
					// true for random selection of m features
					DTC4Point5Node root = tree.buildDT(DecisionTreeApi.dataAsSet, 0, true, m);

					System.out.print("Tree " + t + " ");
					// validation using DFS
					accumulatedResults
							.add(DecisionTreeApi.validation(new HashSet<DataSetTestCase>(validationDataSet), root));
				}

				// choose the tree with the most votes
				for (int v = 0; v < validationDataSet.size(); v++) {
					List<String> votes = new ArrayList<>();
					for (int a = 0; a < accumulatedResults.size(); a++) {
						// for each map find the votes
						votes.add(accumulatedResults.get(a).get(validationDataSet.get(v)));
					}
					// find max vote and add to result
					Map<String, Integer> votesCount = new HashMap<>();
					for (String vote : votes) {
						if (votesCount.containsKey(vote)) {
							votesCount.put(vote, votesCount.get(vote) + 1);
						} else {
							votesCount.put(vote, 1);
						}
					}
					// linear scan to find the highest vote
					int max = Integer.MIN_VALUE;
					String finalVote = "";
					for (Map.Entry<String, Integer> eachMaxVote : votesCount.entrySet()) {
						if (eachMaxVote.getValue() > max) {
							max = eachMaxVote.getValue();
							finalVote = eachMaxVote.getKey();
						}
					}
					finalResultForAFold.put(validationDataSet.get(v), finalVote);
				}

				for (Map.Entry<DataSetTestCase, String> eachFinalResult : finalResultForAFold.entrySet()) {
					DataSetTestCase key = eachFinalResult.getKey();
					String value = eachFinalResult.getValue().trim();
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
				// /TODO remove this
				System.out.println(
						"Final Result Set size: " + finalResultForAFold.size() + " for " + validationDataSet.size());

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
