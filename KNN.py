import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the data
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
users = pd.read_csv('users.csv')

# Preprocess the data
# Remove users with less than 200 ratings and books with less than 100 ratings
min_user_ratings = 200
min_book_ratings = 100

# Filter users and books based on the minimum rating counts
filtered_ratings = ratings[ratings['User-ID'].isin(ratings['User-ID'].value_counts()[ratings['User-ID'].value_counts() >= min_user_ratings].index)]
filtered_ratings = filtered_ratings[filtered_ratings['ISBN'].isin(filtered_ratings['ISBN'].value_counts()[filtered_ratings['ISBN'].value_counts() >= min_book_ratings].index)]

# Create a user-item matrix
pivot_table = filtered_ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')

# Create a sparse matrix for efficiency
book_sparse_matrix = pivot_table.to_sparse()

# Create the model
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model.fit(book_sparse_matrix)

def get_recommends(book_title):
  # Find the index of the book in the pivot table
  book_index = pivot_table.index[pivot_table.index == book_title].tolist()[0]
  
  # Get recommendations
  distances, indices = model.kneighbors(pivot_table.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
  
  # Create a list of recommended books
  recommendations = []
  for i in range(1, len(distances.flatten())):
    book_id = pivot_table.index[indices.flatten()[i]]
    book_name = books.loc[books['ISBN'] == book_id, 'Book-Title'].values[0]
    recommendations.append([book_name, distances.flatten()[i]])
  
  return [book_title, recommendations[1:]]

# Example usage
recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
