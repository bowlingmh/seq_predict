/* fast_ctsnode.c
 *
 * CFFI Code
 *
 * ALPHABET_SIZE, KT_SUM_COUNTS will be filled in dynamically at compile time.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct _ctsnode_t {
  int base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctsnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctsnode_t;

static const double kt_sum_counts = KT_SUM_COUNTS;
static const double kt_start_count = KT_SUM_COUNTS / ALPHABET_SIZE;

double logsumexp2(double a, double b)
{
  if (a > b) {
    return log(1.0 + exp(b - a)) + a;
  } else {
    return log(1.0 + exp(a - b)) + b;
  }
}

ctsnode_t *ctsnode_new(void)
{
  ctsnode_t *self = malloc(sizeof(ctsnode_t));
  for(int i=0; i<ALPHABET_SIZE; i++) self->base_counts[i] = 0;
  self->base_log_prob = log(BASE_PRIOR);
  self->children_log_prob = log(1.0-BASE_PRIOR);
  self->log_prob = 0.0;
  for(int i=0; i<ALPHABET_SIZE; i++) self->children[i] = NULL;
  self->_refcount = 1;

  return self;
}

void ctsnode_free(ctsnode_t *self)
{
  self->_refcount--;
  if (self->_refcount == 0) {
    for(int i=0; i<ALPHABET_SIZE; i++) {
      if (self->children[i]) ctsnode_free(self->children[i]);
    }
    free(self);
  }
}

ctsnode_t *ctsnode_copy(ctsnode_t *self)
{
  ctsnode_t *copy = malloc(sizeof(ctsnode_t));
  (*copy) = (*self);
  copy->_refcount = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) self->children[i]->_refcount++;
  }
  return copy;
}
  
double ctsnode_base_update(ctsnode_t *self, unsigned char symbol)
{
  double sum_counts = kt_sum_counts;
  double lp;

  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];
  lp = log((self->base_counts[symbol] + kt_start_count) / sum_counts);
  self->base_log_prob += lp;
  self->base_counts[symbol] += 1;

  return lp;
}

double ctsnode_update(ctsnode_t *self, unsigned char symbol, unsigned char *context, unsigned int ctxtlen,
		      double log_alpha, double log_blend)
{
  double delta_base_log_prob, orig_log_prob;

  orig_log_prob = self->log_prob;
  delta_base_log_prob = ctsnode_base_update(self, symbol);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctsnode_t *child = self->children[cnext];
    if (!child) child = self->children[cnext] = ctsnode_new();
    else if (child->_refcount > 1) {
      child->_refcount--;
      child = self->children[cnext] = ctsnode_copy(child);
    }

    self->children_log_prob += ctsnode_update(child, symbol, context, ctxtlen-1, log_alpha, log_blend);

    self->log_prob = logsumexp2(self->base_log_prob, self->children_log_prob);
    self->base_log_prob = logsumexp2(log_alpha + self->log_prob, log_blend + self->base_log_prob);
    self->children_log_prob = logsumexp2(log_alpha + self->log_prob, log_blend + self->children_log_prob);
  } else {
    self->log_prob += delta_base_log_prob;
  }

  return self->log_prob - orig_log_prob;
}

double ctsnode_log_predict(ctsnode_t *self, unsigned char symbol, unsigned char *context, unsigned int ctxtlen)
{
  double sum_counts = kt_sum_counts;
  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];

  double base_lp = log((self->base_counts[symbol] + kt_start_count) / sum_counts);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctsnode_t *child =self->children[cnext];
    if (!child) child = self->children[cnext] = ctsnode_new();

    double children_lp = ctsnode_log_predict(child, symbol, context, ctxtlen-1);

    return logsumexp2(self->base_log_prob + base_lp, self->children_log_prob + children_lp) - self->log_prob;
  } else {
    return base_lp;
  }
}

int ctsnode_size(ctsnode_t *self)
{
  int size = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) size += ctsnode_size(self->children[i]);
  }
  return size;
}
