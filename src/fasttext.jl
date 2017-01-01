using ReverseDiff

# Set parameters:
# ngram_range = 2 will add bi-grams features
const NGRAM_RANGE = 1
const MAX_FEATURES = 20000
const MAXLEN = 400
const BATCH_SIZE = 32
const EMBEDDING_DIMS = 50
const NB_EPOCH = 5

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(MAX_FEATURES,
                    EMBEDDING_DIms,
                    input_length=MAXLEN))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
