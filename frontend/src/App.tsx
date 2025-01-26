import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import FileExplorer from "./components/FileExplorer";
import PDFViewer from "./components/PDFViewer";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "./components/ui/resizable";
import Chat from "./components/Chat";
import { useSelectedFile } from "./state/selectedFile";

export default function Page() {
  const { filePath } = useSelectedFile();

  return (
    <div className="flex flex-1 h-screen w-screen">
      <SidebarProvider open={true}>
        <FileExplorer />

        <SidebarInset className="w-full">
          <ResizablePanelGroup
            direction="horizontal"
            // className="min-h-[200px] max-w-md rounded-lg border md:min-w-[450px]"
          >
            {filePath ? (
              <>
                <ResizablePanel defaultSize={50}>
                  <PDFViewer />
                </ResizablePanel>

                <ResizableHandle withHandle />

                <ResizablePanel defaultSize={50}>
                  <Chat />
                </ResizablePanel>
              </>
            ) : (
              <div className="w-full">
                <Chat />
              </div>
            )}
          </ResizablePanelGroup>
        </SidebarInset>
      </SidebarProvider>
    </div>
  );
}
