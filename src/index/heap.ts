/**
 * Binary heap — from scratch.
 *
 * HNSW's inner loop pushes and pops thousands of candidates per query;
 * keeping a sorted array would make every insert O(n). A binary heap
 * gives O(log n) push/pop by maintaining a much weaker invariant than
 * "sorted": every parent outranks its children. The array IS the tree —
 * children of index i live at 2i+1 and 2i+2, so there are no pointers
 * and no allocations beyond the array itself.
 */
export class BinaryHeap<T> {
  private items: T[] = [];

  /** `higherPriority(a, b)` → true if `a` should pop before `b`. */
  constructor(private higherPriority: (a: T, b: T) => boolean) {}

  get size(): number {
    return this.items.length;
  }

  peek(): T | undefined {
    return this.items[0];
  }

  push(item: T): void {
    this.items.push(item);
    // Bubble up: swap with parent while we outrank it.
    let i = this.items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.higherPriority(this.items[i], this.items[parent])) break;
      [this.items[i], this.items[parent]] = [this.items[parent], this.items[i]];
      i = parent;
    }
  }

  pop(): T | undefined {
    if (this.items.length === 0) return undefined;
    const top = this.items[0];
    const last = this.items.pop()!;
    if (this.items.length > 0) {
      this.items[0] = last;
      // Sift down: swap with the higher-priority child until settled.
      let i = 0;
      while (true) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let best = i;
        if (left < this.items.length && this.higherPriority(this.items[left], this.items[best])) best = left;
        if (right < this.items.length && this.higherPriority(this.items[right], this.items[best])) best = right;
        if (best === i) break;
        [this.items[i], this.items[best]] = [this.items[best], this.items[i]];
        i = best;
      }
    }
    return top;
  }

  /** Drain into an array, highest priority first. Empties the heap. */
  drain(): T[] {
    const out: T[] = [];
    while (this.size > 0) out.push(this.pop()!);
    return out;
  }
}
