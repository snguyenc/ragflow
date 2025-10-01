import message from '@/components/ui/message';
import { Spin } from '@/components/ui/spin';
import request from '@/utils/request';
import classNames from 'classnames';
import { useEffect, useState } from 'react';
import { useGetSelectedChunk } from '../../hooks';

interface HtmlPreviewerProps {
  className?: string;
  url: string;
  selectedChunkId?: string;
}

export const HtmlPreviewer: React.FC<HtmlPreviewerProps> = ({
  className,
  url,
  selectedChunkId,
}) => {
  // const url = useGetDocumentUrl();
  const [htmlContent, setHtmlContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const chunk = useGetSelectedChunk(selectedChunkId || '');
  const locationId = chunk?.location_id ?? '';

  console.log(' selectedChunkId', chunk);

  const fetchDocument = async () => {
    setLoading(true);
    const res = await request(url + `?locationId=${locationId}`, {
      method: 'GET',
      responseType: 'blob',
      onError: () => {
        message.error('Document parsing failed');
        console.error('Error loading document:', url);
      },
    });
    try {
      // blob to string
      const reader = new FileReader();
      reader.readAsText(res.data);
      reader.onload = () => {
        setHtmlContent(reader.result as string);
        setLoading(false);
      };
      console.log('file data:', res);
    } catch (err) {
      message.error('Document parsing failed');
      console.error('Error parsing document:', err);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (url) {
      fetchDocument();
    }
  }, [selectedChunkId]);
  return (
    <div
      className={classNames(
        'relative w-full h-full p-4 bg-background-paper border border-border-normal rounded-md',
        className,
      )}
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spin />
        </div>
      )}

      {!loading && (
        <iframe
          src={`data:text/html;charset=utf-8,${encodeURIComponent(htmlContent)}`}
          title="Document Preview"
          className="w-full h-full"
        />
      )}
    </div>
  );
};
