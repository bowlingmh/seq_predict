/* ctwnode.h
 *
 * CFFI Header
 *
 * ALPHABET_SIZE will be filled in dynamically at compile time.
 */

typedef struct _ctwnode_t {
  int base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctwnode_t;

double ctwnode_update(ctwnode_t *node, unsigned char symbol, unsigned char *context, int ctxtlen);
double ctwnode_log_predict(ctwnode_t *node, unsigned char symbol, unsigned char *context, int ctxtlen);
ctwnode_t *ctwnode_new(void);
ctwnode_t *ctwnode_copy(ctwnode_t *self);
void ctwnode_free(ctwnode_t *self);
int ctwnode_size(ctwnode_t *self);
