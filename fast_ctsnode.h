/* fast_ctsnode.h
 *
 * CFFI Header
 *
 * ALPHABET_SIZE will be filled in dynamically at compile time.
 */

typedef struct _ctsnode_t {
  int base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctsnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctsnode_t;

double ctsnode_update(ctsnode_t *node, unsigned char symbol, unsigned char *context, int ctxtlen,
		      double log_alpha, double log_blend);
double ctsnode_log_predict(ctsnode_t *node, unsigned char symbol, unsigned char *context, int ctxtlen);
ctsnode_t *ctsnode_new(void);
ctsnode_t *ctsnode_copy(ctsnode_t *self);
void ctsnode_free(ctsnode_t *self);
int ctsnode_size(ctsnode_t *self);
