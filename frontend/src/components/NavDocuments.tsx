"use client";

import { ChevronRight, type LucideIcon } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import { useSelectedFile } from "@/state/selectedFile";
import { useMessagesStore } from "@/state/messages";

interface DocumentItem {
  title: string;
  url: string;
  icon?: LucideIcon;
  isActive?: boolean;
  items?: DocumentItem[];
}

const NavDocumentSubItem = ({ item }: { item: DocumentItem }) => {
  const { setSelectedFile } = useSelectedFile();
  const { messages, addMessage } = useMessagesStore();

  const handleSelectFile = (filepath: string) => {
    setSelectedFile(filepath);

    if (messages.length === 0) {
      console.log("adding help message");
      addMessage({
        id: "1",
        content: "How can I help you with this document?",
        role: "assistant",
      });
    }
  };

  return (
    <SidebarMenuSubItem>
      {item.items ? (
        <Collapsible asChild className="group/collapsible">
          <div>
            <CollapsibleTrigger asChild>
              <SidebarMenuSubButton>
                {item.icon && <item.icon />}
                <span>{item.title}</span>
                <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
              </SidebarMenuSubButton>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <SidebarMenuSub>
                {item.items.map((subItem) => (
                  <NavDocumentSubItem key={subItem.url} item={subItem} />
                ))}
              </SidebarMenuSub>
            </CollapsibleContent>
          </div>
        </Collapsible>
      ) : (
        <SidebarMenuSubButton
          asChild
          onClick={() => handleSelectFile(item.url)}
        >
          <div>
            {item.icon && <item.icon />}
            <span>{item.title}</span>
          </div>
        </SidebarMenuSubButton>
      )}
    </SidebarMenuSubItem>
  );
};

export function NavDocuments({ items }: { items: DocumentItem[] }) {
  return (
    <SidebarGroup>
      <SidebarGroupLabel>Documents</SidebarGroupLabel>
      <SidebarMenu>
        {items.map((item) => (
          <Collapsible
            key={item.title}
            asChild
            defaultOpen={item.isActive}
            className="group/collapsible"
          >
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton tooltip={item.title}>
                  {item.icon && <item.icon />}
                  <span>{item.title}</span>
                  <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub>
                  {item.items?.map((subItem) => (
                    <NavDocumentSubItem key={subItem.title} item={subItem} />
                  ))}
                </SidebarMenuSub>
              </CollapsibleContent>
            </SidebarMenuItem>
          </Collapsible>
        ))}
      </SidebarMenu>
    </SidebarGroup>
  );
}
