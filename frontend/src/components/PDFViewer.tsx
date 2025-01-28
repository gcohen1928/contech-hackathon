import { useSelectedFile } from "@/state/selectedFile";
import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";

const options = {
  cMapUrl: "/cmaps/",
  standardFontDataUrl: "/standard_fonts/",
};

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

function highlightPattern(text: string, pattern: string) {
  return text.replace(pattern, (value: string) => `<mark>${value}</mark>`);
}

const PDFViewer = () => {
  const [numPages, setNumPages] = useState<number | undefined>(undefined);
  const [searchText, setSearchText] = useState("Region");

  const pageRefs = useRef<(HTMLDivElement | null)[]>([]);

  const { filePath, pageNumber } = useSelectedFile();

  const onDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }): void => {
      setNumPages(numPages);
      pageRefs.current = new Array(numPages).fill(
        null
      ) as (HTMLDivElement | null)[];
    },
    [filePath]
  );

  const textRenderer = useCallback(
    ({ str }: { str: string }) => {
      return highlightPattern(str, searchText);
    },
    [searchText]
  );

  useEffect(() => {
    if (pageNumber && pageRefs.current[pageNumber - 1]) {
      console.log("scrolling to page", pageNumber);
      pageRefs.current[pageNumber - 1]?.scrollIntoView({ behavior: "smooth" });
    }
  }, [pageNumber]);

  return (
    <div className="flex flex-col h-full items-center justify-center p-6">
      <div className="flex flex-grow items-center justify-center overflow-y-auto p-4">
        <Document
          file={filePath}
          onLoadSuccess={onDocumentLoadSuccess}
          className="max-h-full max-w-full"
          options={options}
        >
          {numPages &&
            Array.from(new Array(numPages), (el, index) => (
              <div
                key={`page_${index + 1}`}
                ref={(ref) => (pageRefs.current[index] = ref)}
              >
                <Page
                  pageNumber={index + 1}
                  renderTextLayer={true}
                  renderAnnotationLayer={false}
                  customTextRenderer={textRenderer}
                />
              </div>
            ))}
        </Document>
      </div>
    </div>
  );
};

export default PDFViewer;
