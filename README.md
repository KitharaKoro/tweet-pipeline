# tweet-pipeline

A 4 step Luigi pipeline that cleans the data, transforms the data into a relevant format, performs a multiclass logistic learning algorithm (overkill for the simple model used), and sorts the data based on the trained model.

We explicitly suppress loading all of the data into the memory as much as possible for memory efficiency.
