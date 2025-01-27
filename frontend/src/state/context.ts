import { create } from "zustand";
import { DATABASES, DOCUMENTS } from "./selectedFile";

interface ContextOption {
  value: string;
  label: string;
}

const flattenNestedArray = <T extends { items?: T[] }>(
  arr: T[],
  transform: (item: T) => ContextOption
): ContextOption[] => {
  return arr.reduce<ContextOption[]>((acc, item) => {
    if (item.items && item.items.length > 0) {
      return [...acc, ...flattenNestedArray(item.items, transform)];
    }
    return [...acc, transform(item)];
  }, []);
};

const transform = (item: any): ContextOption => ({
  value: item.url,
  label: item.title,
});

const flattenedDocuments = flattenNestedArray(DOCUMENTS, transform);

export const ContextOptions: ContextOption[] = [
  {
    value: "all",
    label: "All",
  },
  ...flattenedDocuments,
  ...DATABASES.map((db) => ({
    value: db.url,
    label: db.name,
  })),
];

interface ContextState {
  selectedContexts: ContextOption[];
  setSelectedContexts: (contexts: ContextOption[]) => void;
}

export const useContextStore = create<ContextState>((set) => ({
  selectedContexts: ContextOptions, // Start empty instead of with all contexts
  setSelectedContexts: (contexts) => set({ selectedContexts: contexts }),
}));
