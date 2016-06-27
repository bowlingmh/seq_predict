/* ctwnode.c
 *
 * CFFI Code
 *
 * ALPHABET_SIZE will be filled in dynamically at compile time.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct _ctwnode_t {
  int base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctwnode_t;

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

ctwnode_t *ctwnode_new(void)
{
  ctwnode_t *self = malloc(sizeof(ctwnode_t));
  for(int i=0; i<ALPHABET_SIZE; i++) self->base_counts[i] = 0;
  self->base_log_prob = self->children_log_prob = self->log_prob = 0.0;
  for(int i=0; i<ALPHABET_SIZE; i++) self->children[i] = NULL;
  self->_refcount = 1;
  return self;
}

void ctwnode_free(ctwnode_t *self)
{
  self->_refcount--;
  if (self->_refcount == 0) {
    for(int i=0; i<ALPHABET_SIZE; i++) {
      if (self->children[i]) ctwnode_free(self->children[i]);
    }
    free(self);
  }
}

ctwnode_t *ctwnode_copy(ctwnode_t *self)
{
  ctwnode_t *copy = malloc(sizeof(ctwnode_t));
  (*copy) = (*self);
  copy->_refcount = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) self->children[i]->_refcount++;
  }
  return copy;
}
  
void ctwnode_base_update(ctwnode_t *self, unsigned char symbol)
{
  double sum_counts = kt_sum_counts;
  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];
  self->base_log_prob += log((kt_start_count + self->base_counts[symbol]) / sum_counts);
  self->base_counts[symbol] += 1;
}

double ctwnode_update(ctwnode_t *self, unsigned char symbol, unsigned char *context, unsigned int ctxtlen)
{
  double orig_log_prob = self->log_prob;

  ctwnode_base_update(self, symbol);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctwnode_t *child = self->children[cnext];
    if (!child) child = self->children[cnext] = ctwnode_new();
    else if (child->_refcount > 1) {
      child->_refcount--;
      child = self->children[cnext] = ctwnode_copy(child);
    }

    self->children_log_prob += ctwnode_update(child, symbol, context, ctxtlen-1);
    self->log_prob = log(0.5) + logsumexp2(self->base_log_prob, self->children_log_prob);
  } else {
    self->log_prob = self->base_log_prob;
  }

  return self->log_prob - orig_log_prob;
}

double ctwnode_log_predict(ctwnode_t *self, unsigned char symbol, unsigned char *context, unsigned int ctxtlen)
{
  double sum_counts = kt_sum_counts;
  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];

  double base_log_prob = self->base_log_prob + log((kt_start_count + self->base_counts[symbol]) / sum_counts);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctwnode_t *child =self->children[cnext];
    if (!child) child = self->children[cnext] = ctwnode_new();

    double children_log_prob = self->children_log_prob + ctwnode_log_predict(child, symbol, context, ctxtlen-1);

    return log(0.5) + logsumexp2(base_log_prob, children_log_prob) - self->log_prob;
  } else {
    return base_log_prob - self->log_prob;
  }
}

int ctwnode_size(ctwnode_t *self)
{
  int size = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) size += ctwnode_size(self->children[i]);
  }
  return size;
}
