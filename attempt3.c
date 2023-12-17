#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int track;

#define LAMBDA_MAX 1.0
#define LAMBDA_MIN 0.01
#define LAMBDA_STEP 0.01
#define K_FOLDS 10

void least_square_regression(double[][track], double*, int, int, double*);
void ridge_regression(double[][track], double*, int, int, double*);
void lasso_regression(double[][track], double*, int, int, double*);
void add(double*, double*, int);
void subtract(double*, double*, int);
void multiply(double[][track], double*, int, int, double*);
void transpose(double[][track], int, int, double[][5]);
void evaluateMetrics(double*, double*, int, double*, double*);
void standardizeData(double**, double*, int, int);
double kFoldCrossValidation(double**, double*, int, int);
double dotProduct(double*, double*, int);
void crossProduct(double*, double*, double*);
double **allocateMatrix(int, int);
double *allocateVector(int);

int main() {
	freopen("input.txt","r",stdin);
    int observation = 5, parameters = 5;
    double x[observation][parameters];
    double y[observation];
    for (int i = 0; i < observation; i++)
    {
        scanf("%lf",&y[i]);
        for (int j = 0; j < parameters; j++)
        {
            scanf("%lf",&x[i][j]);
        }
    }
    double beta[5];
    double result;

    track = parameters;

    int n;
    printf("Which regression to perform:\n1.Ridge regression\n2.Lasso regression\nEnter your choice:\n");
    scanf("%d",&n);

    if(n==1){
        ridge_regression(x, y, observation, parameters, beta);

        int n1;
        printf("1.See Co-efficients\n2.Predict an output\nEnter your choice: \n");
        scanf("%d",&n1);
        if (n1==1)
        {
            for (int i = 0; i < parameters; i++) {
                printf("%0.2lf\n", beta[i]);
            }
        }
        else
        {
            double x1[] = {8,11,13,4};
            result = 0.0;
            for (int i = 1; i < parameters; i++)
            {
                result += beta[i]*x1[i-1];
            }
            result += beta[0];
            printf("Predicted Y = %0.2lf\n",result);
        }

    }

    else if (n==2)
    {
        lasso_regression(x, y, observation, parameters, beta);

        int n2;
        printf("1.See Co-efficients\n2.Predict an output\nEnter your choice: \n");
        scanf("%d",&n2);
        if (n2==1)
        {
            for (int i = 0; i < parameters; i++) {
                printf("%0.2lf\n", beta[i]);
            }
        }
        else
        {
            double x2[] = {8,11,13,4};
            result = 0.0;
            for (int i = 1; i < parameters; i++)
            {
                result += beta[i]*x2[i-1];
            }
            result += beta[0];
            printf("Predicted Y = %0.2lf\n",result);
        }
    }

    return 0;
}

void add(double* a, double* b, int m) {
    for (int i = 0; i < m; i++) {
        a[i] += b[i];
    }
}

void subtract(double* a, double* b, int m) {
    for (int i = 0; i < m; i++) {
        a[i] -= b[i];
    }
}

void multiply(double x[][track], double* b, int m, int n, double* result) {
    for (int i = 0; i < m; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += x[i][j] * b[j];
        }
    }
}

void transpose(double x[][track], int a, int b, double result[][5]) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            result[j][i] = x[i][j];
        }
    }
}

void least_square_regression(double x[][track], double* y, int observation, int parameters, double* beta) {
    int iterations = 100;
    double learning_rate = 0.01;

    for (int i = 0; i < parameters; i++) {
        beta[i] = 1;
    }

    for (int i = 0; i < iterations; i++) {
        double d_beta[parameters];
        double temp[parameters];
        for (int i = 0; i < parameters; i++)
        {
            d_beta[i] = 0.0;
            temp[i] = 0.0;
        }

        for (int j = 0; j < observation; j++) {
            multiply(x, beta, 1, parameters, temp);
            subtract(temp, y + j, parameters);
            multiply(x + j, temp, parameters, 1, d_beta);
            add(d_beta, temp, parameters);
        }

        for (int j = 0; j < parameters; j++) {
            beta[j] -= learning_rate * d_beta[j] / observation;
        }
    }
}

void ridge_regression(double x[][track], double* y, int observation, int parameters, double* beta) {
    int iterations = 100;
    double learning_rate = 0.01;
    
    double lambda = 1;

    for (int i = 0; i < parameters; i++) {
        beta[i] = 1;
    }

    for (int i = 0; i < iterations; i++) {
        double d_beta[parameters];
        double temp[parameters];
        for (int i = 0; i < parameters; i++)
        {
            d_beta[i] = 0.0;
            temp[i] = 0.0;
        }
        double v[parameters];
        for (int k = 0; k < 5; k++)
        {
            v[k] = lambda*beta[k];
        }

        for (int j = 0; j < observation; j++) {
            multiply(x, beta, 1, parameters, temp);
            subtract(temp, y + j, parameters);
            multiply(x + j, temp, parameters, 1, d_beta);
            add(temp,v,parameters);
            add(d_beta, temp, parameters);
        }

        for (int j = 0; j < parameters; j++) {
            beta[j] -= learning_rate * d_beta[j] / observation;
        }
    }
}

void lasso_regression(double x[][track], double* y, int observation, int parameters, double* beta) {
    int iterations = 100;
    double learning_rate = 0.01;
    double lambda = 1;

    for (int i = 0; i < parameters; i++) {
        beta[i] = 1;
    }

    for (int i = 0; i < iterations; i++) {
        double d_beta[parameters];
        double temp[parameters];
        for (int i = 0; i < parameters; i++)
        {
            d_beta[i] = 0.0;
            temp[i] = 0.0;
        }
        double v[parameters];
        for (int k = 0; k < 5; k++)
        {
            if (beta[k]>0)
            {
                v[k] = lambda*1;
            }
            else if (beta[k]<0)
            {
                v[k] = lambda*-1;
            }
            else
            {
                v[k] = lambda*0;
            }

        }

        for (int j = 0; j < observation; j++) {
            multiply(x, beta, 1, parameters, temp);
            subtract(temp, y + j, parameters);
            multiply(x + j, temp, parameters, 1, d_beta);
            add(temp,v,parameters);
            add(d_beta, temp, parameters);
        }

        for (int j = 0; j < parameters; j++) {
            beta[j] -= learning_rate * d_beta[j] / observation;
        }
    }
}

double dotProduct(double *A, double *B, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += A[i] * B[i];
    }
    return result;
}

void crossProduct(double *A, double *B, double *result) {
    result[0] = A[1] * B[2] - A[2] * B[1];
    result[1] = A[2] * B[0] - A[0] * B[2];
    result[2] = A[0] * B[1] - A[1] * B[0];
}

double **allocateMatrix(int rows, int cols) 
{
    double **matrix = (double **)malloc(rows * sizeof(double *));

    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
    }

    return matrix;
}

double *allocateVector(int size) {
    double *vector = (double *)malloc(size * sizeof(double));

    if (vector == NULL) {
        printf("Memory allocation failed for vector.\n");
        exit(EXIT_FAILURE);
    }

    return vector;
}

void standardizeData(double** x, double* y, int rows, int cols) 
{
    for (int j = 0; j < cols; j++) {
        double mean = 0.0;
        for (int i = 0; i < rows; i++) {
            mean += x[i][j];
        }
        mean /= rows;

        double variation = 0.0;
        for (int i = 0; i < rows; i++) {
            variation += pow(x[i][j] - mean, 2);
        }
        double std_dev = sqrt(variation / rows);

        for (int i = 0; i < rows; i++) {
            x[i][j] = (x[i][j] - mean) / std_dev;
        }
    }
    double mean = 0.0;
    for (int i = 0; i < rows; i++)
    {
        mean += y[i];
    }
    mean /= rows;

    double variation = 0.0;
    for (int i = 0; i < rows; i++)
    {
        variation += pow(y[i]-mean, 2);
    }
    double std_dev = sqrt(variation/rows);

    for (int i = 0; i < rows; i++)
    {
        y[i] = (y[i] - mean)/std_dev;
    }

}

void splitData(double** X, double *y, double **X_train, double *Y_train, double **X_test, double *Y_test,
                int train_size, int test_size, int fold, int cols) {
    int fold_size = train_size / K_FOLDS;

    int test_start = fold * fold_size;
    int test_end = (fold + 1) * fold_size;

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < cols; j++) {
            X_test[i][j] = X[test_start + i][j];
        }
        Y_test[i] = y[test_start + i];
    }

    for (int i = 0; i < test_start; i++) {
        for (int j = 0; j < cols; j++) {
            X_train[i][j] = X[i][j];
        }
        Y_train[i] = y[i];
    }

    for (int i = test_end; i < train_size; i++) {
        for (int j = 0; j < cols; j++) {
            X_train[i - test_size][j] = X[i][j];
        }
        Y_train[i - test_size] = y[i];
    }
}

double kFoldCrossValidation(double **X, double *Y, int rows, int cols) {
    int train_size = rows * 0.8;
    int test_size = rows - train_size;

    double **X_train = allocateMatrix(train_size, cols);
    double **X_test = allocateMatrix(test_size, cols);
    double *Y_train = allocateVector(train_size);
    double *Y_test = allocateVector(test_size);

    double best_lambda = -1;
    double max_r_squared = -INFINITY;

    for (double lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; lambda += LAMBDA_STEP) {
        double r_squared_sum = 0.0;

        for (int fold = 0; fold < K_FOLDS; fold++) {
            splitData(X, Y, X_train, Y_train, X_test, Y_test, train_size, test_size, fold, cols);

            double theta[cols];
            //ridge_regression(X_train, Y_train, cols, train_size, theta);

            double *Y_pred = allocateVector(test_size);
            for (int i = 0; i < test_size; i++) {
                Y_pred[i] = dotProduct(X_test[i], theta, cols);
            }

            double mse, r_squared;
            evaluateMetrics(Y_pred, Y_test, test_size, &mse, &r_squared);

            r_squared_sum += r_squared;
        }

        double avg_r_squared = r_squared_sum / K_FOLDS;

        if (avg_r_squared > max_r_squared) {
            max_r_squared = avg_r_squared;
            best_lambda = lambda;
        }
    }
    return best_lambda;
}

void evaluateMetrics(double *predicted, double *actual, int size, double *mse, double *r_squared)
{
    *mse = 0.0;
    for (int i = 0; i < size; i++) {
        *mse += pow(predicted[i] - actual[i], 2);
    }
    *mse /= size;

    double ssr = 0.0;
    double mean_actual = 0.0;

    for (int i = 0; i < size; i++) {
        mean_actual += actual[i];
    }
    mean_actual /= size;

    for (int i = 0; i < size; i++) {
        ssr += pow(actual[i] - mean_actual, 2);
    }

    *r_squared = 1.0 - (*mse / ssr);
}
