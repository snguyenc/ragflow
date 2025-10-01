import { useTranslate } from '@/hooks/common-hooks';
import { useFormContext } from 'react-hook-form';
import EditTag from './edit-tag';
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from './ui/form';

type BooknameFormFieldProps = {
  name?: string;
};
export function BooknameFormField({
  name = 'parser_config.booknames',
}: BooknameFormFieldProps) {
  const { t } = useTranslate('knowledgeConfiguration');
  const form = useFormContext();

  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => {
        return (
          <FormItem className=" items-center space-y-0 ">
            <div className="flex items-center">
              <FormLabel className="text-sm text-muted-foreground whitespace-nowrap w-1/4">
                <span className="text-red-600">*</span> {'Book Name'}
              </FormLabel>
              <div className="w-3/4">
                <FormControl>
                  <EditTag {...field}></EditTag>
                </FormControl>
              </div>
            </div>
            <div className="flex pt-1">
              <div className="w-1/4"></div>
              <FormMessage />
            </div>
          </FormItem>
        );
      }}
    />
  );
}
