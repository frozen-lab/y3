#define MAX_WORD_LENGTH 128

extern "C" __global__ void compute_edit_distance(const char* word, int word_len,
                                                 const char* word_list,
                                                 int* word_lengths,
                                                 int* distances,
                                                 int num_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    const char* current_word = word_list + idx * MAX_WORD_LENGTH;
    int current_len = word_lengths[idx];

    // Allocate local memory for the DP table
    int dp[MAX_WORD_LENGTH + 1][MAX_WORD_LENGTH + 1];

    // Initialize the DP table
    for (int i = 0; i <= word_len; ++i) dp[i][0] = i;
    for (int j = 0; j <= current_len; ++j) dp[0][j] = j;

    // Compute the edit distance using dynamic programming
    for (int i = 1; i <= word_len; ++i) {
        for (int j = 1; j <= current_len; ++j) {
            if (word[i - 1] == current_word[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] =
                    1 + min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1]));
            }
        }
    }

    // Store the result
    distances[idx] = dp[word_len][current_len];
}
