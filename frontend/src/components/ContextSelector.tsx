import { Check, ChevronsUpDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useEffect, useState } from "react";
import { DATABASES, DOCUMENTS, useSelectedFile } from "@/state/selectedFile";

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

const contextOptions: ContextOption[] = [
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

const ContextSelector = () => {
  const [open, setOpen] = useState(false);
  const [selectedContexts, setSelectedContexts] = useState<ContextOption[]>([]);

  const { filePath } = useSelectedFile();

  useEffect(() => {
    if (filePath) {
      const foundContext = contextOptions.find(
        (context) => context.value === filePath
      );

      if (foundContext) {
        setSelectedContexts([foundContext]);
      }
    }
  }, [filePath]);

  return (
    <div className="w-full flex flex-col">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between h-auto whitespace-normal text-left"
          >
            <span className="mr-2">
              {selectedContexts.length > 0
                ? selectedContexts.some((context) => context.value === "all")
                  ? "All Context Selected"
                  : selectedContexts.map((context) => context.label).join(", ")
                : "Select Context..."}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] max-h-[--radix-popover-content-available-height] p-0">
          <Command>
            <CommandInput placeholder="Search" />
            <CommandList>
              <CommandEmpty>No framework found.</CommandEmpty>
              <CommandGroup>
                {contextOptions.map((context) => (
                  <CommandItem
                    key={context.value}
                    value={context.value}
                    onSelect={() => {
                      if (context.value === "all") {
                        setSelectedContexts(
                          selectedContexts.some(
                            (selected) => selected.value === context.value
                          )
                            ? []
                            : contextOptions
                        );
                      } else {
                        setSelectedContexts(
                          selectedContexts.some(
                            (selected) => selected.value === context.value
                          )
                            ? selectedContexts.filter(
                                (selected) => selected.value !== context.value
                              )
                            : [...selectedContexts, context]
                        );
                      }
                    }}
                  >
                    <Check
                      className={cn(
                        "mr-2 h-4 w-4",
                        selectedContexts.some(
                          (selected) => selected.label === context.label
                        )
                          ? "opacity-100"
                          : "opacity-0"
                      )}
                    />
                    {context.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default ContextSelector;
